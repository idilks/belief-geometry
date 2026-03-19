"""Quick inspection of specific epoch snapshots at higher resolution."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from src.dataset import DEFAULT_COMPONENTS, compute_beliefs_for_sequences
from src.belief_geometry import simplex_to_triangle, three_vertex_colors, simplex_vertices_2d, FACTOR_ANCHORS, get_comp_colors


def get_component_params(snap_dir):
    config_path = Path(snap_dir) / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        if "component_params" in config:
            return [tuple(p) for p in config["component_params"]]
    return DEFAULT_COMPONENTS


def inspect_epoch(epoch, snap_dir, out_dir, component_params=None):
    if component_params is None:
        component_params = get_component_params(snap_dir)

    snap_path = snap_dir / f"activations_epoch{epoch:03d}.npz"
    data = np.load(snap_path)
    comp_ids = data["comp_ids"]
    tokens = data["tokens"]
    final = data["final"]
    N, T, d = final.shape

    beliefs = compute_beliefs_for_sequences(tokens, comp_ids, component_params)
    acts_last = final[:, -1, :]
    beliefs_last = beliefs[:, -1, :]

    K = len(component_params)
    COMP_COLORS = get_comp_colors(K)
    # row 1: 3 fixed panels; row 2: one per component
    n_bottom = max(K, 3)
    fig, axes = plt.subplots(2, n_bottom, figsize=(6 * n_bottom, 11))

    # Row 1: PCA by component, PCA by belief, belief-aligned simplex
    pca = PCA(n_components=2)
    coords = pca.fit_transform(acts_last)

    ax = axes[0, 0]
    for c in range(len(component_params)):
        mask = comp_ids == c
        s_val, r_val = component_params[c]
        ax.scatter(coords[mask, 0], coords[mask, 1], s=8, c=[COMP_COLORS[c % len(COMP_COLORS)]], alpha=0.5,
                   label=f"comp {c} (s={s_val}, r={r_val})")
    ax.set_title(f"PCA — by component (var={pca.explained_variance_ratio_[:2].sum():.2f})")
    ax.legend(fontsize=7, markerscale=3)

    ax = axes[0, 1]
    belief_colors = three_vertex_colors(beliefs_last, alpha=1.5)
    ax.scatter(coords[:, 0], coords[:, 1], s=8, c=belief_colors, alpha=0.5)
    ax.set_title("PCA — by belief state")

    ax = axes[0, 2]
    reg = Ridge(alpha=1.0).fit(acts_last, beliefs_last)
    pred = np.clip(reg.predict(acts_last), 0, None)
    pred /= np.where(pred.sum(1, keepdims=True) > 0, pred.sum(1, keepdims=True), 1.0)
    tri = simplex_to_triangle(pred)
    ax.scatter(tri[:, 0], tri[:, 1], s=8, c=belief_colors, alpha=0.5)
    verts = simplex_vertices_2d()
    for i in range(3):
        j = (i + 1) % 3
        ax.plot([verts[i, 0], verts[j, 0]], [verts[i, 1], verts[j, 1]], "k-", lw=0.8)
    ax.set_aspect("equal")
    ax.set_title("belief-aligned → simplex")
    ax.set_xticks([]); ax.set_yticks([])

    # Row 2: per-component belief simplex vs residual stream
    for c in range(K):
        mask = comp_ids == c
        ax = axes[1, c]
        s_val, r_val = component_params[c]

        b = beliefs[mask].reshape(-1, 3)
        a = final[mask].reshape(-1, d)
        colors = three_vertex_colors(b, alpha=1.5)

        reg_c = Ridge(alpha=1.0).fit(a, b)
        pred_c = np.clip(reg_c.predict(a), 0, None)
        pred_c /= np.where(pred_c.sum(1, keepdims=True) > 0, pred_c.sum(1, keepdims=True), 1.0)
        tri_c = simplex_to_triangle(pred_c)

        ax.scatter(tri_c[:, 0], tri_c[:, 1], s=3, c=colors, alpha=0.4)
        for i in range(3):
            j = (i + 1) % 3
            ax.plot([verts[i, 0], verts[j, 0]], [verts[i, 1], verts[j, 1]], "k-", lw=0.8)
        ax.set_aspect("equal")
        ax.set_title(f"comp {c} (s={s_val}, r={r_val}) — all positions")
        ax.set_xticks([]); ax.set_yticks([])

    # hide unused axes
    for j in range(3, n_bottom):
        axes[0, j].set_visible(False)
    for j in range(K, n_bottom):
        axes[1, j].set_visible(False)

    # R² from linear probe
    X = final.reshape(-1, d)
    Y = beliefs.reshape(-1, 3)
    n = X.shape[0]
    idx = np.random.RandomState(0).permutation(n)
    split = int(0.8 * n)
    reg_full = Ridge(alpha=1.0).fit(X[idx[:split]], Y[idx[:split]])
    Y_pred = reg_full.predict(X[idx[split:]])
    Y_test = Y[idx[split:]]
    r2s = []
    for s in range(3):
        ss_res = ((Y_test[:, s] - Y_pred[:, s]) ** 2).sum()
        ss_tot = ((Y_test[:, s] - Y_test[:, s].mean()) ** 2).sum()
        r2s.append(1 - ss_res / ss_tot)

    fig.suptitle(f"epoch {epoch}  |  R²={[f'{r:.3f}' for r in r2s]}", fontsize=16, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / f"inspect_epoch{epoch:03d}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  epoch {epoch}: R²={[f'{r:.3f}' for r in r2s]}")
    return r2s


def main(snap_dir="checkpoints_long", out_dir="geometry_outputs"):
    snap_dir = Path(snap_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    component_params = get_component_params(snap_dir)
    print(f"component params: {component_params}")

    # find available snapshot epochs
    snaps = sorted(snap_dir.glob("activations_epoch*.npz"))
    available = [int(f.stem.split("epoch")[1]) for f in snaps]
    default_epochs = [5, 20, 30, 50, 100, 300]
    epochs = [e for e in default_epochs if e in available]
    if not epochs:
        epochs = available[:6]

    print(f"inspecting epochs: {epochs}")
    for epoch in epochs:
        inspect_epoch(epoch, snap_dir, out_dir, component_params)


if __name__ == "__main__":
    main()
