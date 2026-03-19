"""
Within-subspace fractal structure analysis.

Does the transformer's per-component activation geometry reproduce the
Mess3 belief fractal? Quantified via Procrustes disparity on
belief-decoded activations, tracked across training epochs.

Key insight: raw PCA of activations picks directions of maximum variance,
which are dominated by non-belief features (position, token identity).
Beliefs explain only ~10% of activation variance. To isolate the belief-
relevant geometry, we first fit a linear decoder (acts -> beliefs), then
project activations onto those decoding directions before comparing to
the ground-truth belief simplex.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from src.dataset import DEFAULT_COMPONENTS, compute_beliefs_for_sequences
from src.belief_geometry import simplex_to_triangle, three_vertex_colors, get_comp_colors_hex
from src.math_utils import procrustes_disparity


def load_snapshot(snap_file):
    data = np.load(snap_file)
    return dict(data)


def get_component_params(snap_dir):
    config_path = Path(snap_dir) / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        if "component_params" in cfg:
            return [tuple(p) for p in cfg["component_params"]]
    return DEFAULT_COMPONENTS


# ---------------------------------------------------------------------------
# Core: decode beliefs from activations, then compare to ground truth
# ---------------------------------------------------------------------------

def decode_beliefs(acts_flat, beliefs_flat, alpha=1.0):
    """
    Fit linear decoder acts -> beliefs, return decoded beliefs on simplex.

    acts_flat: (N, d) activations
    beliefs_flat: (N, 3) true beliefs (used to fit the decoder)
    Returns: (N, 3) decoded beliefs, projected back onto simplex
    """
    reg = Ridge(alpha=alpha).fit(acts_flat, beliefs_flat)
    decoded = reg.predict(acts_flat)
    decoded = np.clip(decoded, 0, 1)
    row_sums = decoded.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    decoded = decoded / row_sums
    return decoded, reg


# ---------------------------------------------------------------------------
# Plot 1: fractal comparison (3 components x 3 columns)
# ---------------------------------------------------------------------------

def plot_fractal_comparison(data, beliefs, component_params, out_dir):
    """
    3 rows (components) x 3 columns:
      1. Ground-truth belief fractal (simplex triangle)
      2. Decoded belief fractal (activations projected onto belief-decoding subspace)
      3. Procrustes-aligned overlay of true vs decoded beliefs

    Uses Ridge regression (acts -> beliefs) to find the belief-relevant subspace,
    rather than PCA which picks maximum-variance directions (dominated by non-belief
    features).
    """
    from scipy.spatial import procrustes as scipy_procrustes

    comp_ids = data["comp_ids"]
    acts = data["final"]  # (N, T, d)
    N, T, d = acts.shape

    K = len(component_params)
    fig, axes = plt.subplots(K, 3, figsize=(14, 4.2 * K + 0.8))
    if K == 1:
        axes = axes[np.newaxis, :]

    disparities = []
    correlations = []

    for c in range(len(component_params)):
        mask = comp_ids == c
        s_val, r_val = component_params[c]

        # beliefs for this component
        b = beliefs[mask]  # (n_c, T, 3)
        b_flat = b.reshape(-1, 3)
        colors = three_vertex_colors(b_flat, alpha=1.5)
        tri_coords = simplex_to_triangle(b_flat)  # (n_c*T, 2)

        # decode beliefs from activations
        a_flat = acts[mask].reshape(-1, d)  # (n_c*T, d)
        decoded, reg = decode_beliefs(a_flat, b_flat)
        decoded_tri = simplex_to_triangle(decoded)
        decoded_colors = three_vertex_colors(decoded, alpha=1.5)

        # per-dim correlation
        from scipy.stats import pearsonr
        corrs = [pearsonr(b_flat[:, i], decoded[:, i])[0] for i in range(3)]
        correlations.append(corrs)

        # Procrustes on simplex coordinates
        n_pts = min(len(tri_coords), len(decoded_tri))
        mtx1, mtx2, disp = scipy_procrustes(tri_coords[:n_pts], decoded_tri[:n_pts])
        disparities.append(disp)

        # col 0: ground-truth belief fractal
        ax = axes[c, 0]
        ax.scatter(tri_coords[:, 0], tri_coords[:, 1], s=1, c=colors, alpha=0.5)
        ax.set_aspect("equal")
        ax.set_title(f"comp {c} (s={s_val}, r={r_val})\ntrue beliefs")
        ax.set_xticks([]); ax.set_yticks([])

        # col 1: decoded belief fractal
        ax = axes[c, 1]
        ax.scatter(decoded_tri[:, 0], decoded_tri[:, 1], s=1, c=decoded_colors, alpha=0.5)
        ax.set_aspect("equal")
        mean_r = np.mean(corrs)
        ax.set_title(f"decoded from activations\nmean r={mean_r:.3f}")
        ax.set_xticks([]); ax.set_yticks([])

        # col 2: Procrustes overlay
        ax = axes[c, 2]
        ax.scatter(mtx1[:, 0], mtx1[:, 1], s=1, c=colors[:n_pts], alpha=0.3, label="true")
        ax.scatter(mtx2[:, 0], mtx2[:, 1], s=1, c="black", alpha=0.15, label="decoded (aligned)")
        ax.set_aspect("equal")
        ax.set_title(f"Procrustes overlay\ndisparity={disp:.4f}")
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=7, markerscale=5)

    plt.suptitle("Within-component fractal structure: true beliefs vs belief-decoded activations", fontsize=13)
    plt.tight_layout()
    out_path = out_dir / "fractal_comparison.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")
    print(f"  disparities: {[f'{d:.4f}' for d in disparities]}")
    for c, corrs in enumerate(correlations):
        print(f"  comp {c} per-dim r: [{', '.join(f'{r:.3f}' for r in corrs)}]")
    return disparities


# ---------------------------------------------------------------------------
# Plot 2: fractal emergence across training
# ---------------------------------------------------------------------------

def plot_fractal_evolution(snap_dir, component_params, out_dir):
    """
    Procrustes disparity (belief-decoded) vs training epoch for each component.
    Shows when the fractal structure emerges.

    At each epoch, fits a fresh Ridge decoder from that epoch's activations
    to beliefs, then measures Procrustes disparity between decoded and true
    belief simplex coordinates.
    """
    snap_dir = Path(snap_dir)
    snap_files = sorted(snap_dir.glob("activations_epoch*.npz"))
    if not snap_files:
        print(f"no snapshots found in {snap_dir}")
        return

    epochs = []
    disp_by_comp = {c: [] for c in range(len(component_params))}
    corr_by_comp = {c: [] for c in range(len(component_params))}

    for sf in snap_files:
        epoch = int(sf.stem.split("epoch")[1])
        data = load_snapshot(sf)
        comp_ids = data["comp_ids"]
        tokens = data["tokens"]
        acts = data["final"]
        N, T, d = acts.shape

        beliefs = compute_beliefs_for_sequences(tokens, comp_ids, component_params)

        epochs.append(epoch)
        for c in range(len(component_params)):
            mask = comp_ids == c
            b_flat = beliefs[mask].reshape(-1, 3)
            tri_coords = simplex_to_triangle(b_flat)
            a_flat = acts[mask].reshape(-1, d)

            if a_flat.shape[0] < 10:
                disp_by_comp[c].append(np.nan)
                corr_by_comp[c].append(np.nan)
                continue

            decoded, _ = decode_beliefs(a_flat, b_flat)
            decoded_tri = simplex_to_triangle(decoded)
            n_pts = min(len(tri_coords), len(decoded_tri))
            disp = procrustes_disparity(tri_coords[:n_pts], decoded_tri[:n_pts])
            disp_by_comp[c].append(disp)

            # mean correlation across belief dimensions
            from scipy.stats import pearsonr
            mean_r = np.mean([pearsonr(b_flat[:, i], decoded[:, i])[0] for i in range(3)])
            corr_by_comp[c].append(mean_r)

    # plot: two panels — disparity and correlation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = get_comp_colors_hex(len(component_params))

    for c in range(len(component_params)):
        s_val, r_val = component_params[c]
        label = f"comp {c} (s={s_val}, r={r_val})"
        ax1.plot(epochs, disp_by_comp[c], "o-", color=colors[c], markersize=4, label=label)
        ax2.plot(epochs, corr_by_comp[c], "o-", color=colors[c], markersize=4, label=label)

    ax1.set_xlabel("epoch")
    ax1.set_ylabel("Procrustes disparity (lower = better)")
    ax1.set_title("Fractal match: belief-decoded disparity")
    ax1.legend(fontsize=8)
    ax1.set_ylim(bottom=0)

    ax2.set_xlabel("epoch")
    ax2.set_ylabel("mean Pearson r (higher = better)")
    ax2.set_title("Belief decoding quality")
    ax2.legend(fontsize=8)
    ax2.set_ylim(-0.1, 1.05)

    for ax in (ax1, ax2):
        if len(epochs) > 1:
            ax.set_xscale("log")

    plt.tight_layout()
    out_path = out_dir / "fractal_evolution.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")
    print(f"  epochs: {epochs}")
    for c in range(len(component_params)):
        d_vals = disp_by_comp[c]
        r_vals = corr_by_comp[c]
        print(f"  comp {c}: disp {d_vals[0]:.4f}->{d_vals[-1]:.4f}, r {r_vals[0]:.4f}->{r_vals[-1]:.4f}")


# ---------------------------------------------------------------------------
# Plot 3: layer-by-layer fractal quality (reload model for all layers)
# ---------------------------------------------------------------------------

def plot_fractal_by_layer(snap_dir, component_params, out_dir, epoch=None):
    """
    Procrustes disparity at each layer for one epoch.
    Reloads model weights to get intermediate layer activations.
    """
    import torch
    from src.transformer import Mess3Transformer
    from src.dataset import Mess3NonErgodicDataset

    snap_dir = Path(snap_dir)
    config_path = snap_dir / "config.json"
    with open(config_path) as f:
        cfg = json.load(f)

    # find model file
    if epoch is None:
        # use best or final
        model_file = snap_dir / "model_final.pt"
        if not model_file.exists():
            pts = sorted(snap_dir.glob("model_epoch*.pt"))
            if pts:
                model_file = pts[-1]
            else:
                print("no model files found")
                return
        epoch = int(model_file.stem.split("epoch")[1]) if "epoch" in model_file.stem else -1
    else:
        model_file = snap_dir / f"model_epoch{epoch:03d}.pt"
        if not model_file.exists():
            print(f"model file {model_file} not found")
            return

    context_len = cfg.get("seq_len", 16) - 1
    model = Mess3Transformer(
        vocab_size=3,
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"],
        max_len=context_len,
        dropout=0.0,
    )
    model.load_state_dict(torch.load(model_file, map_location="cpu", weights_only=True))
    model.eval()

    # generate data
    dataset = Mess3NonErgodicDataset(
        component_params=component_params,
        num_sequences_per_component=500,
        seq_len=cfg.get("seq_len", 16),
        seed=99,
    )

    # extract all layers
    per_comp = 500
    indices = []
    for c in range(len(component_params)):
        comp_indices = np.where(dataset.component_ids == c)[0][:per_comp]
        indices.append(comp_indices)
    indices = np.concatenate(indices)

    subset = torch.utils.data.Subset(dataset, indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=512, shuffle=False)

    all_residuals = {}
    all_comp_ids = []
    all_tokens = []

    with torch.no_grad():
        for input_ids, _targets, comp_ids in loader:
            residuals = model.extract_residual_stream(input_ids)
            for key, val in residuals.items():
                if key not in all_residuals:
                    all_residuals[key] = []
                all_residuals[key].append(val.numpy())
            all_comp_ids.append(comp_ids.numpy() if isinstance(comp_ids, torch.Tensor) else np.array(comp_ids))
            all_tokens.append(input_ids.numpy())

    for key in all_residuals:
        all_residuals[key] = np.concatenate(all_residuals[key])
    comp_ids_arr = np.concatenate(all_comp_ids)
    tokens_arr = np.concatenate(all_tokens)

    beliefs = compute_beliefs_for_sequences(tokens_arr, comp_ids_arr, component_params)

    layer_names = sorted(all_residuals.keys(), key=lambda x: (0 if x == "embed" else 2 if x == "final" else 1, x))

    # compute disparities and correlations per layer per component
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = get_comp_colors_hex(len(component_params))

    for c in range(len(component_params)):
        mask = comp_ids_arr == c
        b_flat = beliefs[mask].reshape(-1, 3)
        tri_coords = simplex_to_triangle(b_flat)
        disps = []
        mean_rs = []
        for lname in layer_names:
            a = all_residuals[lname]
            d_dim = a.shape[-1]
            a_flat = a[mask].reshape(-1, d_dim)
            decoded, _ = decode_beliefs(a_flat, b_flat)
            decoded_tri = simplex_to_triangle(decoded)
            n_pts = min(len(tri_coords), len(decoded_tri))
            disp = procrustes_disparity(tri_coords[:n_pts], decoded_tri[:n_pts])
            disps.append(disp)

            from scipy.stats import pearsonr
            mean_r = np.mean([pearsonr(b_flat[:, i], decoded[:, i])[0] for i in range(3)])
            mean_rs.append(mean_r)

        s_val, r_val = component_params[c]
        label = f"comp {c} (s={s_val}, r={r_val})"
        ax1.plot(range(len(layer_names)), disps, "o-", color=colors[c], markersize=6, label=label)
        ax2.plot(range(len(layer_names)), mean_rs, "o-", color=colors[c], markersize=6, label=label)

    for ax in (ax1, ax2):
        ax.set_xticks(range(len(layer_names)))
        ax.set_xticklabels(layer_names, rotation=30)
        ax.legend(fontsize=8)

    ax1.set_ylabel("Procrustes disparity (lower = better)")
    ax1.set_title(f"Belief-decoded fractal match by layer (epoch {epoch})")
    ax1.set_ylim(bottom=0)

    ax2.set_ylabel("mean Pearson r (higher = better)")
    ax2.set_title(f"Belief decoding quality by layer (epoch {epoch})")
    ax2.set_ylim(-0.1, 1.05)

    plt.tight_layout()
    out_path = out_dir / "fractal_by_layer.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


# ---------------------------------------------------------------------------
def main(snap_dir="checkpoints_long", out_dir="geometry_outputs"):
    snap_dir = Path(snap_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    component_params = get_component_params(snap_dir)
    print(f"component params: {component_params}")

    # find latest snapshot
    snaps = sorted(snap_dir.glob("activations_epoch*.npz"))
    if not snaps:
        print(f"no snapshots in {snap_dir}")
        return
    snap_file = snaps[-1]
    print(f"loading {snap_file}...")
    data = load_snapshot(snap_file)
    N = data["comp_ids"].shape[0]
    T = data["tokens"].shape[1]
    print(f"  N={N}, T={T}")

    print("computing beliefs...")
    beliefs = compute_beliefs_for_sequences(data["tokens"], data["comp_ids"], component_params)

    print("\n--- fractal comparison ---")
    plot_fractal_comparison(data, beliefs, component_params, out_dir)

    print("\n--- fractal evolution ---")
    plot_fractal_evolution(snap_dir, component_params, out_dir)

    print("\n--- fractal by layer ---")
    plot_fractal_by_layer(snap_dir, component_params, out_dir)

    print("\ndone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snap-dir", default="checkpoints_long")
    parser.add_argument("--out-dir", default="geometry_outputs")
    args = parser.parse_args()
    main(args.snap_dir, args.out_dir)
