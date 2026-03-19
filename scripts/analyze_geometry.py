"""
Residual stream geometry analysis.

Loads trained model activations, computes ground-truth HMM beliefs for the
same sequences, and visualizes the geometry at each layer.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA

from src.dataset import DEFAULT_COMPONENTS, compute_beliefs_for_sequences
from src.belief_geometry import simplex_to_triangle, three_vertex_colors, FACTOR_ANCHORS, get_comp_colors


def load_config(snap_dir):
    config_path = Path(snap_dir) / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return None


def get_component_params(snap_dir):
    config = load_config(snap_dir)
    if config and "component_params" in config:
        return [tuple(p) for p in config["component_params"]]
    return DEFAULT_COMPONENTS


def load_activations(path="checkpoints/activations.npz"):
    data = np.load(path)
    return dict(data)


def pca_activations(acts, n_components=3):
    """PCA on flattened (N*T, d_model) activations. Returns coords + fitted PCA."""
    orig_shape = acts.shape  # (N, T, d)
    flat = acts.reshape(-1, acts.shape[-1])
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(flat)
    return coords, pca


# ---------------------------------------------------------------------------
# Plot 1: PCA at each layer, colored by component
# ---------------------------------------------------------------------------
def plot_component_clusters(data, out_dir, component_params=None):
    """PCA scatter at each layer, colored by component ID."""
    if component_params is None:
        component_params = DEFAULT_COMPONENTS
    comp_ids = data["comp_ids"]  # (N,)
    layer_names = ["embed", "layer_0", "layer_1", "final"]
    # only use layers that exist in the data
    layer_names = [n for n in layer_names if n in data]
    colors_map = get_comp_colors(len(component_params))

    ncols = len(layer_names)
    nrows = (ncols + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 5 * nrows))
    axes = axes.ravel() if ncols > 1 else [axes]

    for idx, name in enumerate(layer_names):
        acts = data[name]  # (N, T, d)
        N, T, d = acts.shape
        last_pos = acts[:, -1, :]  # (N, d)
        pca = PCA(n_components=2)
        coords = pca.fit_transform(last_pos)

        ax = axes[idx]
        for c in range(len(component_params)):
            mask = comp_ids == c
            s_val, r_val = component_params[c]
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       s=4, c=[colors_map[c % len(colors_map)]], alpha=0.5,
                       label=f"comp {c} (s={s_val}, r={r_val})")

        ax.set_title(f"{name}  (pos={T-1}, var={pca.explained_variance_ratio_[:2].sum():.2f})")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(fontsize=7, markerscale=3)

    # hide unused axes
    for idx in range(len(layer_names), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Residual stream PCA (last position) — colored by component", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_dir / "component_clusters.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_dir / 'component_clusters.png'}")


# ---------------------------------------------------------------------------
# Plot 2: Position evolution — how geometry changes with context
# ---------------------------------------------------------------------------
def plot_position_evolution(data, out_dir, component_params=None):
    """PCA at final layer across positions, showing how clusters tighten."""
    if component_params is None:
        component_params = DEFAULT_COMPONENTS
    comp_ids = data["comp_ids"]
    acts = data["final"]  # (N, T, d)
    N, T, d = acts.shape
    colors_map = get_comp_colors(len(component_params))

    positions = [0, 2, 6, T - 1]
    all_vecs = np.concatenate([acts[:, p, :] for p in positions], axis=0)
    pca = PCA(n_components=2)
    pca.fit(all_vecs)

    fig, axes = plt.subplots(1, len(positions), figsize=(4 * len(positions), 4))

    for i, pos in enumerate(positions):
        coords = pca.transform(acts[:, pos, :])
        ax = axes[i]
        for c in range(len(component_params)):
            mask = comp_ids == c
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       s=4, c=[colors_map[c % len(colors_map)]], alpha=0.4)
        ax.set_title(f"pos {pos}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_xlim(coords[:, 0].min() - 0.5, coords[:, 0].max() + 0.5)
        ax.set_ylim(coords[:, 1].min() - 0.5, coords[:, 1].max() + 0.5)

    plt.suptitle("Final layer — component separation by position", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_dir / "position_evolution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_dir / 'position_evolution.png'}")


# ---------------------------------------------------------------------------
# Plot 3: Residual stream vs ground-truth beliefs
# ---------------------------------------------------------------------------
def plot_belief_comparison(data, beliefs, out_dir, component_params=None):
    if component_params is None:
        component_params = DEFAULT_COMPONENTS
    comp_ids = data["comp_ids"]
    acts = data["final"]  # (N, T, d)
    N, T, d = acts.shape

    K = len(component_params)
    fig, axes = plt.subplots(K, 2, figsize=(10, 4.5 * K + 0.5))
    if K == 1:
        axes = axes[np.newaxis, :]

    for c in range(len(component_params)):
        mask = comp_ids == c
        s_val, r_val = component_params[c]

        b = beliefs[mask]  # (n_c, T, 3)
        b_flat = b.reshape(-1, 3)
        colors = three_vertex_colors(b_flat, alpha=1.5)
        tri_coords = simplex_to_triangle(b_flat)

        ax_belief = axes[c, 0]
        ax_belief.scatter(tri_coords[:, 0], tri_coords[:, 1], s=2, c=colors, alpha=0.5)
        ax_belief.set_aspect("equal")
        ax_belief.set_title(f"comp {c} (s={s_val}, r={r_val}) — HMM beliefs")
        ax_belief.set_xticks([])
        ax_belief.set_yticks([])

        a = acts[mask]  # (n_c, T, d)
        a_flat = a.reshape(-1, d)
        pca = PCA(n_components=2)
        coords = pca.fit_transform(a_flat)

        ax_res = axes[c, 1]
        ax_res.scatter(coords[:, 0], coords[:, 1], s=2, c=colors, alpha=0.5)
        ax_res.set_title(f"comp {c} — residual stream PCA (final)")
        ax_res.set_xlabel("PC1")
        ax_res.set_ylabel("PC2")

    plt.suptitle("HMM belief geometry vs residual stream geometry", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_dir / "belief_vs_residual.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_dir / 'belief_vs_residual.png'}")


# ---------------------------------------------------------------------------
# Plot 4: Linear probe — can we decode beliefs from activations?
# ---------------------------------------------------------------------------
def plot_belief_correlation(data, beliefs, out_dir, component_params=None):
    from sklearn.linear_model import Ridge

    if component_params is None:
        component_params = DEFAULT_COMPONENTS
    comp_ids = data["comp_ids"]
    acts = data["final"]  # (N, T, d)
    N, T, d = acts.shape

    X = acts.reshape(-1, d)
    Y = beliefs.reshape(-1, 3)

    n = X.shape[0]
    idx = np.random.RandomState(0).permutation(n)
    split = int(0.8 * n)
    X_tr, X_te = X[idx[:split]], X[idx[split:]]
    Y_tr, Y_te = Y[idx[:split]], Y[idx[split:]]

    reg = Ridge(alpha=1.0).fit(X_tr, Y_tr)
    Y_pred = reg.predict(X_te)
    Y_pred = np.clip(Y_pred, 0, 1)

    r2s = []
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    state_names = ["state 0", "state 1", "state 2"]
    state_colors = FACTOR_ANCHORS

    for s in range(3):
        ax = axes[s]
        ss_res = np.sum((Y_te[:, s] - Y_pred[:, s]) ** 2)
        ss_tot = np.sum((Y_te[:, s] - Y_te[:, s].mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        r2s.append(r2)

        n_plot = min(5000, len(Y_te))
        ax.scatter(Y_te[:n_plot, s], Y_pred[:n_plot, s], s=1, alpha=0.3, c=[state_colors[s]])
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("true belief")
        ax.set_ylabel("predicted belief")
        ax.set_title(f"{state_names[s]}  R²={r2:.3f}")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")

    plt.suptitle("Linear probe: decoding HMM beliefs from final-layer activations", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_dir / "belief_linear_probe.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_dir / 'belief_linear_probe.png'}  R²={r2s}")
    return r2s


# ---------------------------------------------------------------------------
# Plot 5: explained variance across layers
# ---------------------------------------------------------------------------
def plot_dimensionality(data, out_dir):
    layer_names = ["embed", "layer_0", "layer_1", "final"]
    layer_names = [n for n in layer_names if n in data]

    fig, ax = plt.subplots(figsize=(8, 5))

    for name in layer_names:
        acts = data[name]  # (N, T, d)
        flat = acts[:, -1, :]  # last position
        pca = PCA(n_components=min(20, flat.shape[1]))
        pca.fit(flat)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        ax.plot(range(1, len(cumvar) + 1), cumvar, marker="o", markersize=3, label=name)

    ax.axhline(0.95, color="gray", ls="--", lw=0.8, label="95%")
    ax.set_xlabel("# PCA components")
    ax.set_ylabel("cumulative explained variance")
    ax.set_title("Effective dimensionality across layers (last position)")
    ax.legend()
    ax.set_xlim(1, 20)
    ax.set_ylim(0, 1.02)

    plt.tight_layout()
    fig.savefig(out_dir / "dimensionality.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_dir / 'dimensionality.png'}")


# ---------------------------------------------------------------------------
def main(snap_dir="checkpoints_long", out_dir="geometry_outputs", snap_file=None):
    snap_dir = Path(snap_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    component_params = get_component_params(snap_dir)
    print(f"component params: {component_params}")

    if snap_file is None:
        snap_file = snap_dir / "activations.npz"
        if not snap_file.exists():
            # try latest snapshot
            snaps = sorted(snap_dir.glob("activations_epoch*.npz"))
            if snaps:
                snap_file = snaps[-1]
            else:
                print(f"no activation files found in {snap_dir}")
                return

    print(f"loading activations from {snap_file}...")
    data = load_activations(snap_file)
    print(f"  N={data['comp_ids'].shape[0]}, T={data['tokens'].shape[1]}, d={data['embed'].shape[2]}")

    print("computing ground-truth beliefs...")
    beliefs = compute_beliefs_for_sequences(data["tokens"], data["comp_ids"], component_params)
    print(f"  beliefs shape: {beliefs.shape}")

    print("\n--- generating plots ---\n")
    plot_component_clusters(data, out_dir, component_params)
    plot_position_evolution(data, out_dir, component_params)
    plot_belief_comparison(data, beliefs, out_dir, component_params)
    r2s = plot_belief_correlation(data, beliefs, out_dir, component_params)
    plot_dimensionality(data, out_dir)

    print("\ndone. all plots in", out_dir)
    return r2s


if __name__ == "__main__":
    main()
