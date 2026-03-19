"""
Mechanistic dissection of forward filtering in the trained transformer.

Extracts attention patterns, runs head/MLP ablations, and measures the
differential impact on component classification vs belief tracking.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge, LogisticRegression

from src.dataset import (
    Mess3NonErgodicDataset,
    DEFAULT_COMPONENTS,
    compute_beliefs_for_sequences,
)
from src.transformer import Mess3Transformer


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def load_model_and_data(checkpoint_dir, model_file=None):
    """Load model and build matching eval dataset."""
    checkpoint_dir = Path(checkpoint_dir)
    with open(checkpoint_dir / "config.json") as f:
        config = json.load(f)

    component_params = [tuple(p) for p in config["component_params"]]

    model = Mess3Transformer(
        vocab_size=3,
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        d_ff=config["d_ff"],
        max_len=config["seq_len"] - 1,
        dropout=config.get("dropout", 0.0),
    )
    if model_file is None:
        model_file = checkpoint_dir / "model_best.pt"
    state = torch.load(model_file, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # eval dataset — use a separate seed from training
    dataset = Mess3NonErgodicDataset(
        component_params=component_params,
        num_sequences_per_component=1000,
        seq_len=config["seq_len"],
        seed=999,
    )

    return model, dataset, config, component_params


def get_eval_batch(dataset, max_n=3000):
    """Get all eval data as tensors."""
    per_comp = max_n // dataset.num_components
    indices = []
    for c in range(dataset.num_components):
        comp_idx = np.where(dataset.component_ids == c)[0][:per_comp]
        indices.append(comp_idx)
    indices = np.concatenate(indices)

    input_ids_list, targets_list, comp_ids_list = [], [], []
    for i in indices:
        inp, tgt, cid = dataset[i]
        input_ids_list.append(inp)
        targets_list.append(tgt)
        comp_ids_list.append(cid)

    input_ids = torch.stack(input_ids_list)    # (N, T)
    targets = torch.stack(targets_list)        # (N, T)
    comp_ids = np.array(comp_ids_list)         # (N,)
    tokens = input_ids.numpy()                 # (N, T)

    return input_ids, targets, comp_ids, tokens


# ---------------------------------------------------------------------------
# Probes: component classification + belief R²
# ---------------------------------------------------------------------------

def train_probes(activations, comp_ids, beliefs):
    """
    Train component classifier and belief regressor on activations.

    activations: (N, T, d) — uses last position
    comp_ids: (N,)
    beliefs: (N, T, 3) — uses last position

    Returns (comp_acc, belief_r2) on held-out 20%.
    """
    N = activations.shape[0]
    X = activations[:, -1, :]  # last position
    y_comp = comp_ids
    y_belief = beliefs[:, -1, :]

    # train/test split
    rng = np.random.RandomState(42)
    idx = rng.permutation(N)
    split = int(0.8 * N)
    X_tr, X_te = X[idx[:split]], X[idx[split:]]
    yc_tr, yc_te = y_comp[idx[:split]], y_comp[idx[split:]]
    yb_tr, yb_te = y_belief[idx[:split]], y_belief[idx[split:]]

    # component classification
    clf = LogisticRegression(max_iter=1000, random_state=0).fit(X_tr, yc_tr)
    comp_acc = clf.score(X_te, yc_te)

    # belief R²
    reg = Ridge(alpha=1.0).fit(X_tr, yb_tr)
    yb_pred = reg.predict(X_te)
    ss_res = np.sum((yb_te - yb_pred) ** 2)
    ss_tot = np.sum((yb_te - yb_te.mean(axis=0)) ** 2)
    belief_r2 = 1 - ss_res / ss_tot

    return comp_acc, belief_r2


# ---------------------------------------------------------------------------
# A. Attention pattern extraction + visualization
# ---------------------------------------------------------------------------

def extract_all_attention(model, input_ids, batch_size=512):
    """Extract attention weights for all data, batched."""
    N = input_ids.shape[0]
    n_layers = model.n_layers
    all_att = [[] for _ in range(n_layers)]

    for start in range(0, N, batch_size):
        batch = input_ids[start:start + batch_size]
        att_list = model.extract_attention_weights(batch)
        for layer_idx in range(n_layers):
            all_att[layer_idx].append(att_list[layer_idx].numpy())

    return [np.concatenate(layer_atts, axis=0) for layer_atts in all_att]


def plot_attention_patterns(att_by_layer, n_heads, out_dir):
    """2×4 grid of average attention heatmaps (mean over all sequences)."""
    n_layers = len(att_by_layer)
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(3.5 * n_heads, 3.5 * n_layers))
    if n_layers == 1:
        axes = axes[np.newaxis, :]

    for layer_idx in range(n_layers):
        avg_att = att_by_layer[layer_idx].mean(axis=0)  # (H, T, T)
        for h in range(n_heads):
            ax = axes[layer_idx, h]
            im = ax.imshow(avg_att[h], cmap="Blues", vmin=0, aspect="auto")
            ax.set_title(f"L{layer_idx}H{h}", fontsize=10)
            if h == 0:
                ax.set_ylabel("query pos")
            if layer_idx == n_layers - 1:
                ax.set_xlabel("key pos")
            ax.tick_params(labelsize=7)

    fig.colorbar(im, ax=axes, shrink=0.6, label="attention weight")
    plt.suptitle("Average attention patterns per head", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_dir / "attention_patterns.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_dir / 'attention_patterns.png'}")


def plot_attention_by_token(att_by_layer, input_ids, n_heads, out_dir):
    """
    Attention patterns conditioned on query token identity.
    For each head, show 3 sub-heatmaps (one per token at query position).
    """
    tokens_np = input_ids.numpy() if isinstance(input_ids, torch.Tensor) else input_ids
    n_layers = len(att_by_layer)
    n_tokens = 3

    fig, axes = plt.subplots(n_layers * n_heads, n_tokens,
                             figsize=(3 * n_tokens, 2.2 * n_layers * n_heads))
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)

    row = 0
    for layer_idx in range(n_layers):
        att = att_by_layer[layer_idx]  # (N, H, T, T)
        N, H, T, _ = att.shape
        for h in range(n_heads):
            for tok in range(n_tokens):
                ax = axes[row, tok]
                # for each query position, select sequences where that position has this token
                avg_pattern = np.zeros((T, T))
                count = np.zeros(T)
                for pos in range(T):
                    mask = tokens_np[:, pos] == tok
                    if mask.sum() > 0:
                        avg_pattern[pos] = att[mask, h, pos, :].mean(axis=0)
                        count[pos] = mask.sum()

                im = ax.imshow(avg_pattern, cmap="Blues", vmin=0, aspect="auto")
                if tok == 0:
                    ax.set_ylabel(f"L{layer_idx}H{h}", fontsize=9)
                if row == 0:
                    ax.set_title(f"query tok={tok}", fontsize=9)
                ax.tick_params(labelsize=6)
            row += 1

    plt.suptitle("Attention conditioned on query token identity", fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(out_dir / "attention_by_token.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_dir / 'attention_by_token.png'}")


def plot_attention_by_component(att_by_layer, comp_ids, n_heads, out_dir):
    """Per-head attention patterns conditioned on component identity."""
    n_layers = len(att_by_layer)
    n_comps = len(np.unique(comp_ids))

    fig, axes = plt.subplots(n_layers * n_heads, n_comps,
                             figsize=(3 * n_comps, 2.2 * n_layers * n_heads))
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)

    row = 0
    for layer_idx in range(n_layers):
        att = att_by_layer[layer_idx]  # (N, H, T, T)
        for h in range(n_heads):
            for c in range(n_comps):
                ax = axes[row, c]
                mask = comp_ids == c
                avg = att[mask, h].mean(axis=0)  # (T, T)
                im = ax.imshow(avg, cmap="Blues", vmin=0, aspect="auto")
                if c == 0:
                    ax.set_ylabel(f"L{layer_idx}H{h}", fontsize=9)
                if row == 0:
                    ax.set_title(f"comp {c}", fontsize=9)
                ax.tick_params(labelsize=6)
            row += 1

    plt.suptitle("Attention conditioned on component identity", fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(out_dir / "attention_by_component.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_dir / 'attention_by_component.png'}")


# ---------------------------------------------------------------------------
# B. Head ablation
# ---------------------------------------------------------------------------

def run_ablation_experiment(model, input_ids, targets, comp_ids, beliefs, config):
    """
    Ablate each head individually and measure impact on:
    - component classification accuracy
    - belief R²
    - next-token cross-entropy loss

    Also does layer-wise ablation (all heads in a layer) and MLP ablation.
    """
    n_layers = config["n_layers"]
    n_heads = config["n_heads"]
    criterion = nn.CrossEntropyLoss()

    # --- baseline (no ablation) ---
    logits_base, acts_base = model.forward_with_ablation(input_ids)
    base_loss = criterion(logits_base.reshape(-1, 3), targets.reshape(-1)).item()
    base_comp_acc, base_belief_r2 = train_probes(acts_base.numpy(), comp_ids, beliefs)
    print(f"baseline: loss={base_loss:.4f}, comp_acc={base_comp_acc:.3f}, belief_R2={base_belief_r2:.3f}")

    results = {
        "baseline": {"loss": base_loss, "comp_acc": base_comp_acc, "belief_r2": base_belief_r2}
    }

    # --- individual head ablation ---
    for layer_idx in range(n_layers):
        for h in range(n_heads):
            label = f"L{layer_idx}H{h}"
            logits, acts = model.forward_with_ablation(
                input_ids, ablate_heads={layer_idx: [h]}
            )
            loss = criterion(logits.reshape(-1, 3), targets.reshape(-1)).item()
            comp_acc, belief_r2 = train_probes(acts.numpy(), comp_ids, beliefs)
            results[label] = {"loss": loss, "comp_acc": comp_acc, "belief_r2": belief_r2}
            print(f"  ablate {label}: loss={loss:.4f} (d={loss - base_loss:+.4f}), "
                  f"comp_acc={comp_acc:.3f} (d={comp_acc - base_comp_acc:+.3f}), "
                  f"belief_R2={belief_r2:.3f} (d={belief_r2 - base_belief_r2:+.3f})")

    # --- full layer ablation ---
    for layer_idx in range(n_layers):
        label = f"layer_{layer_idx}_all_heads"
        logits, acts = model.forward_with_ablation(
            input_ids, ablate_heads={layer_idx: list(range(n_heads))}
        )
        loss = criterion(logits.reshape(-1, 3), targets.reshape(-1)).item()
        comp_acc, belief_r2 = train_probes(acts.numpy(), comp_ids, beliefs)
        results[label] = {"loss": loss, "comp_acc": comp_acc, "belief_r2": belief_r2}
        print(f"  ablate {label}: loss={loss:.4f} (d={loss - base_loss:+.4f}), "
              f"comp_acc={comp_acc:.3f} (d={comp_acc - base_comp_acc:+.3f}), "
              f"belief_R2={belief_r2:.3f} (d={belief_r2 - base_belief_r2:+.3f})")

    # --- MLP ablation ---
    for layer_idx in range(n_layers):
        label = f"layer_{layer_idx}_mlp"
        logits, acts = model.forward_with_ablation(
            input_ids, ablate_mlps=[layer_idx]
        )
        loss = criterion(logits.reshape(-1, 3), targets.reshape(-1)).item()
        comp_acc, belief_r2 = train_probes(acts.numpy(), comp_ids, beliefs)
        results[label] = {"loss": loss, "comp_acc": comp_acc, "belief_r2": belief_r2}
        print(f"  ablate {label}: loss={loss:.4f} (d={loss - base_loss:+.4f}), "
              f"comp_acc={comp_acc:.3f} (d={comp_acc - base_comp_acc:+.3f}), "
              f"belief_R2={belief_r2:.3f} (d={belief_r2 - base_belief_r2:+.3f})")

    return results


def plot_ablation_table(results, out_dir):
    """Bar chart of Δ metrics for each individual head ablation."""
    baseline = results["baseline"]

    # collect individual heads
    head_labels = [k for k in results if k.startswith("L")]
    head_labels.sort()

    delta_loss = [results[h]["loss"] - baseline["loss"] for h in head_labels]
    delta_comp = [results[h]["comp_acc"] - baseline["comp_acc"] for h in head_labels]
    delta_r2 = [results[h]["belief_r2"] - baseline["belief_r2"] for h in head_labels]

    x = np.arange(len(head_labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, delta_loss, width, label="Δ loss (↑ = worse)", color="#d62728")
    ax.bar(x, delta_comp, width, label="Δ comp_acc (↓ = worse)", color="#1f77b4")
    ax.bar(x + width, delta_r2, width, label="Δ belief_R² (↓ = worse)", color="#2ca02c")

    ax.set_xticks(x)
    ax.set_xticklabels(head_labels)
    ax.set_ylabel("Δ from baseline")
    ax.set_title(f"Per-head ablation (baseline: loss={baseline['loss']:.4f}, "
                 f"acc={baseline['comp_acc']:.3f}, R²={baseline['belief_r2']:.3f})")
    ax.legend()
    ax.axhline(0, color="black", lw=0.5)

    # add vertical separator between layers
    n_per_layer = len([h for h in head_labels if h.startswith("L0")])
    if n_per_layer < len(head_labels):
        ax.axvline(n_per_layer - 0.5, color="gray", ls="--", lw=0.8)

    plt.tight_layout()
    fig.savefig(out_dir / "ablation_table.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_dir / 'ablation_table.png'}")


def plot_ablation_layer(results, out_dir):
    """Layer-level ablation: full-layer heads + MLP, side by side."""
    baseline = results["baseline"]

    layer_keys = sorted([k for k in results if k.endswith("_all_heads") or k.endswith("_mlp")])
    if not layer_keys:
        return

    labels = []
    delta_loss = []
    delta_comp = []
    delta_r2 = []

    for k in layer_keys:
        # clean up label
        lbl = k.replace("layer_", "L").replace("_all_heads", " attn").replace("_mlp", " MLP")
        labels.append(lbl)
        delta_loss.append(results[k]["loss"] - baseline["loss"])
        delta_comp.append(results[k]["comp_acc"] - baseline["comp_acc"])
        delta_r2.append(results[k]["belief_r2"] - baseline["belief_r2"])

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, delta_loss, width, label="Δ loss (↑ = worse)", color="#d62728")
    ax.bar(x, delta_comp, width, label="Δ comp_acc (↓ = worse)", color="#1f77b4")
    ax.bar(x + width, delta_r2, width, label="Δ belief_R² (↓ = worse)", color="#2ca02c")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Δ from baseline")
    ax.set_title("Layer-level ablation: attention vs MLP")
    ax.legend()
    ax.axhline(0, color="black", lw=0.5)

    plt.tight_layout()
    fig.savefig(out_dir / "ablation_layer.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_dir / 'ablation_layer.png'}")


# ---------------------------------------------------------------------------
# Attention entropy summary (quantifies broad vs local)
# ---------------------------------------------------------------------------

def plot_attention_entropy(att_by_layer, n_heads, out_dir):
    """
    Per-head attention entropy as a function of query position.
    High entropy = broad attention (pooling). Low entropy = focused/local.
    """
    n_layers = len(att_by_layer)

    fig, axes = plt.subplots(1, n_layers, figsize=(6 * n_layers, 4), squeeze=False)

    for layer_idx in range(n_layers):
        att = att_by_layer[layer_idx]  # (N, H, T, T)
        N, H, T, _ = att.shape
        ax = axes[0, layer_idx]

        for h in range(H):
            # entropy at each query position, averaged over sequences
            # only over valid (non-masked) positions
            entropies = np.zeros(T)
            for pos in range(T):
                probs = att[:, h, pos, :pos + 1]  # (N, pos+1) — valid keys
                # add small epsilon for numerical stability
                log_probs = np.log(probs + 1e-10)
                ent = -(probs * log_probs).sum(axis=-1).mean()
                # normalize by max possible entropy at this position
                max_ent = np.log(pos + 1)
                entropies[pos] = ent / max_ent if max_ent > 0 else 0

            ax.plot(range(T), entropies, marker="o", markersize=3, label=f"H{h}")

        ax.set_xlabel("query position")
        ax.set_ylabel("normalized entropy")
        ax.set_title(f"Layer {layer_idx} — attention entropy")
        ax.legend()
        ax.set_ylim(0, 1.1)

    plt.tight_layout()
    fig.savefig(out_dir / "attention_entropy.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_dir / 'attention_entropy.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(checkpoint_dir="checkpoints_2L_simple_reg", out_dir=None, model_file=None):
    checkpoint_dir = Path(checkpoint_dir)
    if out_dir is None:
        out_dir = Path(f"mechanism_{checkpoint_dir.name}")
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    print(f"checkpoint: {checkpoint_dir}")
    print(f"output: {out_dir}")

    # load model + data
    model, dataset, config, component_params = load_model_and_data(checkpoint_dir, model_file=model_file)
    n_heads = config["n_heads"]
    n_layers = config["n_layers"]
    print(f"model: {n_layers}L, {n_heads}H, d={config['d_model']}")

    input_ids, targets, comp_ids, tokens = get_eval_batch(dataset)
    N, T = input_ids.shape
    print(f"eval data: {N} sequences, {T} positions")

    # ground truth beliefs
    beliefs = compute_beliefs_for_sequences(tokens, comp_ids, component_params)
    print(f"beliefs: {beliefs.shape}")

    # --- A. attention patterns ---
    print("\n--- extracting attention weights ---")
    att_by_layer = extract_all_attention(model, input_ids)
    for i, att in enumerate(att_by_layer):
        print(f"  layer {i}: {att.shape}")

    print("\n--- plotting attention patterns ---")
    plot_attention_patterns(att_by_layer, n_heads, out_dir)
    plot_attention_by_token(att_by_layer, input_ids, n_heads, out_dir)
    plot_attention_by_component(att_by_layer, comp_ids, n_heads, out_dir)
    plot_attention_entropy(att_by_layer, n_heads, out_dir)

    # --- B + C. ablation ---
    print("\n--- running ablation experiments ---")
    results = run_ablation_experiment(model, input_ids, targets, comp_ids, beliefs, config)

    print("\n--- plotting ablation results ---")
    plot_ablation_table(results, out_dir)
    plot_ablation_layer(results, out_dir)

    # save raw results
    with open(out_dir / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {out_dir / 'ablation_results.json'}")

    print("\ndone. all outputs in", out_dir)
    return results


def ablation_over_training(checkpoint_dir="checkpoints_2L_simple_reg", epochs=None, out_dir=None):
    """
    Run ablation at multiple training epochs and plot how head/layer importance
    evolves over training.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if out_dir is None:
        out_dir = Path(f"mechanism_{checkpoint_dir.name}")
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    with open(checkpoint_dir / "config.json") as f:
        config = json.load(f)

    component_params = [tuple(p) for p in config["component_params"]]
    n_heads = config["n_heads"]
    n_layers = config["n_layers"]

    # find available epoch checkpoints
    if epochs is None:
        epoch_files = sorted(checkpoint_dir.glob("model_epoch*.pt"))
        epochs = []
        for ef in epoch_files:
            ep = int(ef.stem.split("epoch")[1])
            epochs.append(ep)
    print(f"will run ablation at epochs: {epochs}")

    # build eval data once
    dataset = Mess3NonErgodicDataset(
        component_params=component_params,
        num_sequences_per_component=1000,
        seq_len=config["seq_len"],
        seed=999,
    )
    input_ids, targets, comp_ids, tokens = get_eval_batch(dataset)
    beliefs = compute_beliefs_for_sequences(tokens, comp_ids, component_params)
    criterion = nn.CrossEntropyLoss()

    # collect results per epoch
    all_epoch_results = {}
    for ep in epochs:
        model_file = checkpoint_dir / f"model_epoch{ep:03d}.pt"
        if not model_file.exists():
            print(f"  skipping epoch {ep} (no checkpoint)")
            continue

        print(f"\n=== epoch {ep} ===")
        model = Mess3Transformer(
            vocab_size=3,
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            n_layers=config["n_layers"],
            d_ff=config["d_ff"],
            max_len=config["seq_len"] - 1,
            dropout=config.get("dropout", 0.0),
        )
        state = torch.load(model_file, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()

        results = run_ablation_experiment(model, input_ids, targets, comp_ids, beliefs, config)
        all_epoch_results[ep] = results

    # --- plot: ablation effect over training ---
    plot_ablation_over_training(all_epoch_results, n_layers, n_heads, out_dir)

    # save raw
    serializable = {str(k): v for k, v in all_epoch_results.items()}
    with open(out_dir / "ablation_over_training.json", "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nsaved {out_dir / 'ablation_over_training.json'}")


def plot_ablation_over_training(all_epoch_results, n_layers, n_heads, out_dir):
    """
    Plot how ablation effects change over training epochs.
    Three subplots: d_loss, d_comp_acc, d_belief_r2 over epochs.
    Lines for each individual head + layer-level ablations.
    """
    epochs = sorted(all_epoch_results.keys())

    # collect head labels and layer labels
    head_labels = [f"L{l}H{h}" for l in range(n_layers) for h in range(n_heads)]
    layer_attn_labels = [f"layer_{l}_all_heads" for l in range(n_layers)]
    layer_mlp_labels = [f"layer_{l}_mlp" for l in range(n_layers)]

    metrics = ["loss", "comp_acc", "belief_r2"]
    metric_titles = ["d loss (+ = worse)", "d comp_acc (- = worse)", "d belief_R2 (- = worse)"]
    signs = [1, -1, -1]  # for coloring: positive = bad

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for mi, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes[mi]

        # individual heads: thin lines
        for head in head_labels:
            vals = []
            for ep in epochs:
                base = all_epoch_results[ep]["baseline"][metric]
                ablated = all_epoch_results[ep][head][metric]
                vals.append(ablated - base)
            layer_idx = int(head[1])
            color = plt.cm.Set1(layer_idx / max(n_layers - 1, 1))
            ax.plot(epochs, vals, lw=0.8, alpha=0.5, color=color)

        # layer-level attention: thick lines
        for label in layer_attn_labels:
            vals = []
            for ep in epochs:
                base = all_epoch_results[ep]["baseline"][metric]
                ablated = all_epoch_results[ep][label][metric]
                vals.append(ablated - base)
            layer_idx = int(label.split("_")[1])
            color = plt.cm.Set1(layer_idx / max(n_layers - 1, 1))
            lbl = f"L{layer_idx} attn (all)"
            ax.plot(epochs, vals, lw=2.5, color=color, label=lbl, marker="o", markersize=4)

        # layer-level MLP: thick dashed
        for label in layer_mlp_labels:
            vals = []
            for ep in epochs:
                base = all_epoch_results[ep]["baseline"][metric]
                ablated = all_epoch_results[ep][label][metric]
                vals.append(ablated - base)
            layer_idx = int(label.split("_")[1])
            color = plt.cm.Set1(layer_idx / max(n_layers - 1, 1))
            lbl = f"L{layer_idx} MLP"
            ax.plot(epochs, vals, lw=2.5, ls="--", color=color, label=lbl, marker="s", markersize=4)

        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("epoch")
        ax.set_ylabel(f"d {metric}")
        ax.set_title(title)
        ax.legend(fontsize=7)

    plt.suptitle("Ablation effects over training", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_dir / "ablation_over_training.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_dir / 'ablation_over_training.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints_2L_simple_reg")
    parser.add_argument("--out", default=None)
    parser.add_argument("--model-file", default=None, help="specific model .pt file")
    parser.add_argument("--over-training", action="store_true",
                        help="run ablation at all available epoch checkpoints")
    parser.add_argument("--epochs", nargs="*", type=int, default=None,
                        help="specific epochs for --over-training")
    args = parser.parse_args()

    if args.over_training:
        ablation_over_training(
            checkpoint_dir=args.checkpoint,
            epochs=args.epochs,
            out_dir=args.out,
        )
    else:
        main(
            checkpoint_dir=args.checkpoint,
            out_dir=args.out,
            model_file=args.model_file,
        )
