"""
Generate GIF of residual stream geometry evolving over training.

Two views per frame:
  - Left: PCA of final-layer activations (last position), colored by component
  - Right: Belief-aligned projection (using linear probe weights), colored by HMM belief

Also generates a static belief-aligned projection plot for the final checkpoint.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
import io

from src.dataset import DEFAULT_COMPONENTS, compute_beliefs_for_sequences
from src.belief_geometry import simplex_to_triangle, three_vertex_colors, FACTOR_ANCHORS, simplex_vertices_2d, get_comp_colors


def get_component_params(snap_dir):
    config_path = Path(snap_dir) / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        if "component_params" in config:
            return [tuple(p) for p in config["component_params"]]
    return DEFAULT_COMPONENTS


def belief_aligned_projection(activations, beliefs):
    reg = Ridge(alpha=1.0).fit(activations, beliefs)
    pred = reg.predict(activations)
    pred = np.clip(pred, 0, None)
    sums = pred.sum(axis=1, keepdims=True)
    sums = np.where(sums > 0, sums, 1.0)
    pred /= sums
    return pred


def make_frame(snap_path, epoch, beliefs_cache, component_params, pca_global=None):
    """Generate one frame (as PIL Image) for the given snapshot."""
    data = np.load(snap_path)
    comp_ids = data["comp_ids"]
    tokens = data["tokens"]
    final = data["final"]  # (N, T, d)
    N, T, d = final.shape

    cache_key = (snap_path.name,)
    if cache_key not in beliefs_cache:
        beliefs_cache[cache_key] = compute_beliefs_for_sequences(tokens, comp_ids, component_params)
    beliefs = beliefs_cache[cache_key]

    acts_last = final[:, -1, :]       # (N, d)
    beliefs_last = beliefs[:, -1, :]  # (N, 3)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: PCA colored by component
    if pca_global is not None:
        coords = pca_global.transform(acts_last)
    else:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(acts_last)

    comp_colors = get_comp_colors(len(component_params))
    for c in range(len(component_params)):
        mask = comp_ids == c
        s_val, r_val = component_params[c]
        ax1.scatter(coords[mask, 0], coords[mask, 1],
                    s=6, c=[comp_colors[c]], alpha=0.5,
                    label=f"comp {c}")
    ax1.set_title("PCA — by component")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.legend(fontsize=7, markerscale=3, loc="upper right")

    # Panel 2: PCA colored by belief state
    belief_colors = three_vertex_colors(beliefs_last, alpha=1.5)
    ax2.scatter(coords[:, 0], coords[:, 1], s=6, c=belief_colors, alpha=0.5)
    ax2.set_title("PCA — by belief state")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")

    # Panel 3: belief-aligned projection → simplex
    pred_beliefs = belief_aligned_projection(acts_last, beliefs_last)
    tri_coords = simplex_to_triangle(pred_beliefs)
    belief_colors2 = three_vertex_colors(beliefs_last, alpha=1.5)
    ax3.scatter(tri_coords[:, 0], tri_coords[:, 1], s=6, c=belief_colors2, alpha=0.5)
    verts = simplex_vertices_2d()
    for i in range(3):
        j = (i + 1) % 3
        ax3.plot([verts[i, 0], verts[j, 0]], [verts[i, 1], verts[j, 1]], "k-", lw=0.8)
    ax3.set_aspect("equal")
    ax3.set_title("belief-aligned projection → simplex")
    ax3.set_xticks([])
    ax3.set_yticks([])

    fig.suptitle(f"epoch {epoch}", fontsize=16, fontweight="bold")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def main(snap_dir="checkpoints_long", out_dir="geometry_outputs"):
    snap_dir = Path(snap_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    component_params = get_component_params(snap_dir)
    print(f"component params: {component_params}")

    snap_files = sorted(snap_dir.glob("activations_epoch*.npz"))
    if not snap_files:
        print("no snapshots found in", snap_dir)
        return

    epochs = []
    for f in snap_files:
        e = int(f.stem.split("epoch")[1])
        epochs.append(e)
    print(f"found {len(snap_files)} snapshots: epochs {epochs}")

    # fit a global PCA on the final snapshot for consistent axes
    last_data = np.load(snap_files[-1])
    final_acts = last_data["final"][:, -1, :]
    pca_global = PCA(n_components=2).fit(final_acts)

    beliefs_cache = {}
    frames = []
    for snap_path, epoch in zip(snap_files, epochs):
        print(f"  rendering epoch {epoch}...")
        frame = make_frame(snap_path, epoch, beliefs_cache, component_params, pca_global)
        frames.append(frame)

    gif_path = out_dir / "geometry_evolution.gif"
    durations = [800] * len(frames)
    durations[-1] = 3000
    frames[0].save(
        gif_path, save_all=True, append_images=frames[1:],
        duration=durations, loop=0,
    )
    print(f"\nsaved {gif_path}")

    frames[-1].save(out_dir / "geometry_final.png")
    print(f"saved {out_dir / 'geometry_final.png'}")


if __name__ == "__main__":
    main()
