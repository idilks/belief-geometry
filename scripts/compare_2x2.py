"""
2x2 comparison figure: model size x parameter regime.

Loads the best-epoch snapshot from each condition, produces:
1. 2x2 grid of belief-aligned simplex projections with R² annotated
2. Training curves overlaid (4 conditions)
3. Summary table printed to stdout
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

from src.dataset import compute_beliefs_for_sequences
from src.belief_geometry import simplex_to_triangle, three_vertex_colors, simplex_vertices_2d


CONDITIONS = {
    "2L_simple": {"ckpt": "checkpoints_2L_simple", "geo": "geometry_2L_simple"},
    "2L_rich":   {"ckpt": "checkpoints_2L_rich",   "geo": "geometry_2L_rich"},
    "3L_simple": {"ckpt": "checkpoints_long",       "geo": "geometry_3L_simple"},
    "3L_rich":   {"ckpt": "checkpoints_3L_rich",    "geo": "geometry_3L_rich"},
}

GRID_POS = {
    "2L_simple": (0, 0),
    "2L_rich":   (0, 1),
    "3L_simple": (1, 0),
    "3L_rich":   (1, 1),
}

LABELS = {
    "2L_simple": "2-layer, simple params",
    "2L_rich":   "2-layer, rich params",
    "3L_simple": "3-layer, simple params",
    "3L_rich":   "3-layer, rich params",
}


def load_condition(name):
    """Load config, best-epoch activations, and history for one condition."""
    info = CONDITIONS[name]
    ckpt_dir = Path(info["ckpt"])

    config_path = ckpt_dir / "config.json"
    if not config_path.exists():
        print(f"  {name}: no config.json found, skipping")
        return None
    with open(config_path) as f:
        config = json.load(f)

    component_params = [tuple(p) for p in config["component_params"]]
    best_epoch = config.get("best_epoch", None)

    # find best-epoch snapshot
    snap_file = None
    if best_epoch:
        candidate = ckpt_dir / f"activations_epoch{best_epoch:03d}.npz"
        if candidate.exists():
            snap_file = candidate

    # fallback: nearest available snapshot to best_epoch
    if snap_file is None:
        snaps = sorted(ckpt_dir.glob("activations_epoch*.npz"))
        if snaps and best_epoch:
            snap_epochs = [int(s.stem.split("epoch")[1]) for s in snaps]
            nearest_idx = min(range(len(snap_epochs)), key=lambda i: abs(snap_epochs[i] - best_epoch))
            snap_file = snaps[nearest_idx]
            actual_epoch = snap_epochs[nearest_idx]
            print(f"  {name}: best_epoch={best_epoch} not snapshotted, using nearest epoch {actual_epoch}")
            best_epoch = actual_epoch
        elif snaps:
            snap_file = snaps[-1]
            best_epoch = int(snap_file.stem.split("epoch")[1])

    if snap_file is None:
        print(f"  {name}: no activation snapshots found, skipping")
        return None

    data = np.load(snap_file)

    # history
    history = None
    hist_path = ckpt_dir / "history.npz"
    if hist_path.exists():
        history = dict(np.load(hist_path))

    return {
        "config": config,
        "component_params": component_params,
        "best_epoch": best_epoch,
        "data": dict(data),
        "history": history,
        "snap_file": snap_file,
    }


def compute_r2(data, component_params):
    """Compute mean R² from linear probe on final-layer activations."""
    final = data["final"]
    tokens = data["tokens"]
    comp_ids = data["comp_ids"]
    N, T, d = final.shape

    beliefs = compute_beliefs_for_sequences(tokens, comp_ids, component_params)

    X = final.reshape(-1, d)
    Y = beliefs.reshape(-1, 3)
    n = X.shape[0]
    idx = np.random.RandomState(0).permutation(n)
    split = int(0.8 * n)

    reg = Ridge(alpha=1.0).fit(X[idx[:split]], Y[idx[:split]])
    Y_pred = reg.predict(X[idx[split:]])
    Y_test = Y[idx[split:]]

    r2s = []
    for s in range(3):
        ss_res = ((Y_test[:, s] - Y_pred[:, s]) ** 2).sum()
        ss_tot = ((Y_test[:, s] - Y_test[:, s].mean()) ** 2).sum()
        r2s.append(1 - ss_res / ss_tot if ss_tot > 0 else 0.0)

    return r2s, np.mean(r2s), beliefs


def make_simplex_panel(ax, data, beliefs, component_params, title, r2_mean):
    """Draw belief-aligned simplex projection on given axis."""
    final = data["final"]
    N, T, d = final.shape
    acts_last = final[:, -1, :]
    beliefs_last = beliefs[:, -1, :]

    belief_colors = three_vertex_colors(beliefs_last, alpha=1.5)

    reg = Ridge(alpha=1.0).fit(acts_last, beliefs_last)
    pred = np.clip(reg.predict(acts_last), 0, None)
    sums = pred.sum(axis=1, keepdims=True)
    sums = np.where(sums > 0, sums, 1.0)
    pred /= sums
    tri = simplex_to_triangle(pred)

    ax.scatter(tri[:, 0], tri[:, 1], s=6, c=belief_colors, alpha=0.5)

    verts = simplex_vertices_2d()
    for i in range(3):
        j = (i + 1) % 3
        ax.plot([verts[i, 0], verts[j, 0]], [verts[i, 1], verts[j, 1]], "k-", lw=0.8)

    ax.set_aspect("equal")
    ax.set_title(f"{title}\nR²={r2_mean:.3f}", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])

    # annotate component params
    params_str = ", ".join(f"({s},{r})" for s, r in component_params)
    ax.text(0.5, -0.05, params_str, transform=ax.transAxes,
            ha="center", va="top", fontsize=7, color="gray")


def main():
    out_dir = Path("geometry_2x2")
    out_dir.mkdir(exist_ok=True)

    # load all conditions
    results = {}
    for name in CONDITIONS:
        print(f"loading {name}...")
        result = load_condition(name)
        if result is not None:
            r2s, r2_mean, beliefs = compute_r2(result["data"], result["component_params"])
            result["r2s"] = r2s
            result["r2_mean"] = r2_mean
            result["beliefs"] = beliefs
            results[name] = result

    if not results:
        print("no conditions loaded!")
        return

    # --- Figure 1: 2x2 simplex grid ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    for name, (row, col) in GRID_POS.items():
        ax = axes[row, col]
        if name in results:
            r = results[name]
            epoch_label = f"epoch {r['best_epoch']}" if r['best_epoch'] else ""
            title = f"{LABELS[name]} ({epoch_label})"
            make_simplex_panel(ax, r["data"], r["beliefs"], r["component_params"], title, r["r2_mean"])
        else:
            ax.set_visible(False)

    fig.suptitle("Belief-aligned simplex projections — 2x2 comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "simplex_2x2.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_dir / 'simplex_2x2.png'}")

    # --- Figure 2: training curves ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"2L_simple": "C0", "2L_rich": "C1", "3L_simple": "C2", "3L_rich": "C3"}

    for name in CONDITIONS:
        if name not in results or results[name]["history"] is None:
            continue
        h = results[name]["history"]
        epochs = np.arange(1, len(h["train_loss"]) + 1)
        c = colors[name]
        ax1.plot(epochs, h["train_loss"], c=c, alpha=0.4, lw=0.8)
        ax1.plot(epochs, h["val_loss"], c=c, lw=1.5, label=LABELS[name])
        if "best_epoch" in results[name]["config"]:
            be = results[name]["config"]["best_epoch"]
            if be <= len(h["val_loss"]):
                ax1.axvline(be, c=c, ls="--", lw=0.8, alpha=0.5)

    ax1.axhline(np.log(3), color="gray", ls=":", lw=1, label="uniform baseline")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_title("Training curves (solid=val, faint=train)")
    ax1.legend(fontsize=8)
    ax1.set_ylim(bottom=0.9)

    for name in CONDITIONS:
        if name not in results or results[name]["history"] is None:
            continue
        h = results[name]["history"]
        epochs = np.arange(1, len(h["val_acc"]) + 1)
        ax2.plot(epochs, h["val_acc"], c=colors[name], lw=1.5, label=LABELS[name])

    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax2.set_title("Validation accuracy")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_dir / "training_curves_2x2.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_dir / 'training_curves_2x2.png'}")

    # --- Summary table ---
    # determine max number of R² columns across conditions
    max_states = max((len(r["r2s"]) for r in results.values()), default=3)
    r2_headers = " ".join(f"{'R²['+str(i)+']':>7}" for i in range(max_states))
    print(f"\n{'='*80}")
    print(f"{'Condition':<20} {'Layers':>6} {'Params':>8} {'BestEp':>7} {'ValLoss':>8} {'R²mean':>8} {r2_headers}")
    print(f"{'-'*80}")
    for name in ["2L_simple", "2L_rich", "3L_simple", "3L_rich"]:
        if name not in results:
            print(f"{name:<20} {'--':>6}")
            continue
        r = results[name]
        cfg = r["config"]
        n_params = cfg.get("n_params", "?")
        r2_vals = " ".join(f"{v:>7.3f}" for v in r["r2s"])
        print(
            f"{name:<20} {cfg['n_layers']:>6} {n_params:>8} "
            f"{r['best_epoch']:>7} {cfg.get('best_val_loss', 0):>8.4f} "
            f"{r['r2_mean']:>8.3f} {r2_vals}"
        )
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
