#!/usr/bin/env python3
"""
Mock belief-geometry script for a 3-state 'mess3-like' HMM.

This is a simplified, stand-alone analogue of experiments/figure_generation/figure1/scripts/mess3_beliefs.py,
but specialized to our single-factor Mess3HMM.

It:
  - builds a Mess3HMM
  - generates many belief states p(z_t | y_{1:t}) over 3 hidden states
  - embeds beliefs into:
      * a 2D equilateral triangle (simplex embedding)
      * a 3D space using PCA on "joint" belief-like vectors
  - colors points based on belief mass
  - renders Matplotlib figures

Usage:
    python mess3_belief_geometry.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA

from mess3_hmm import Mess3HMM, generate_beliefs_batch


# -----------------------------------------------------------------------------
# Simplex / geometry utilities
# -----------------------------------------------------------------------------

def simplex_to_triangle(beliefs: np.ndarray) -> np.ndarray:
    """
    Map 3-simplex coordinates (p0, p1, p2) to an equilateral triangle in 2D.

    Args:
        beliefs: array of shape (N, 3), rows sum to 1.

    Returns:
        coords: array of shape (N, 2).
    """
    x = beliefs[:, 1] + 0.5 * beliefs[:, 2]
    y = (np.sqrt(3) / 2.0) * beliefs[:, 2]
    return np.stack([x, y], axis=1)


def simplex_vertices_2d() -> np.ndarray:
    """Coordinates of the triangle vertices (images of one-hot basis)."""
    eye3 = np.eye(3)
    return simplex_to_triangle(eye3)


# -----------------------------------------------------------------------------
# Color utilities: interpolate between three anchor colors
# -----------------------------------------------------------------------------

# anchor colors for states 0, 1, 2 (RGB in [0,1])
FACTOR_ANCHORS = np.array([
    [0.89, 0.10, 0.11],  # state 0: red-ish
    [0.99, 0.55, 0.00],  # state 1: orange
    [0.12, 0.47, 0.71],  # state 2: blue
])


def three_vertex_colors(beliefs: np.ndarray, anchors: np.ndarray = FACTOR_ANCHORS,
                        alpha: float = 1.5) -> np.ndarray:
    """
    Blend colors based on a 3-state belief distribution.

    Args:
        beliefs: (N, 3) belief vectors.
        anchors: (3, 3) RGB colors for the 3 pure states.
        alpha:   emphasize confident distributions (alpha>1).

    Returns:
        colors: (N, 3) RGB in [0,1].
    """
    b = np.clip(beliefs, 0.0, 1.0)
    # emphasize large probabilities
    w = b ** alpha
    w_sum = w.sum(axis=1, keepdims=True)
    w_sum = np.where(w_sum > 0, w_sum, 1.0)
    w /= w_sum
    return w @ anchors


# -----------------------------------------------------------------------------
# 3D embedding of beliefs
# -----------------------------------------------------------------------------

def pca_embed_beliefs(beliefs: np.ndarray, random_state: int = 0) -> tuple[np.ndarray, PCA]:
    """
    Embed 3-state beliefs into 3D via PCA on a simple "joint-like" representation.

    Since we only have one factor, we can either:
      - run PCA directly on beliefs (shape (N,3)), or
      - create a synthetic 9D "joint" (outer product of beliefs with itself) and
        then do PCA, which mimics the 9D joint in the real repo.

    Here we do the "joint" trick to look more like their Figure-1 pipeline.

    Returns:
        coords_3d: (N, 3)
        pca:       fitted PCA object
    """
    # synthetic joint: 3x3 outer product -> flatten to 9D
    joint = np.einsum("ni,nj->nij", beliefs, beliefs).reshape(-1, 9)

    pca = PCA(n_components=3, random_state=random_state)
    coords_3d = pca.fit_transform(joint)
    return coords_3d, pca


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------

def style_3d_axes(ax, elev: float = 30, azim: float = -60):
    ax.view_init(elev=elev, azim=azim)
    if hasattr(ax, "set_proj_type"):
        ax.set_proj_type("persp")
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)


def set_equal_3d_limits(ax, coords: np.ndarray, margin: float = 0.05):
    """
    Set equal x,y,z limits around coords, with a small margin.
    """
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = (maxs - mins).max()
    half = 0.5 * span * (1.0 + margin)
    x_min, x_max = center[0] - half, center[0] + half
    y_min, y_max = center[1] - half, center[1] + half
    z_min, z_max = center[2] - half, center[2] + half
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)


# -----------------------------------------------------------------------------
# Main figure: factor simplex in 2D + 3D embedding
# -----------------------------------------------------------------------------

def make_belief_geometry_figure(
    beliefs: np.ndarray,
    out_dir: Path,
    title_prefix: str = "Mess3 Belief Geometry",
):
    """
    Create a figure showing:
      - beliefs in 3D (PCA embedding of synthetic joint)
      - beliefs in 2D (triangle / simplex)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3D embedding via PCA
    coords_3d, pca = pca_embed_beliefs(beliefs, random_state=0)
    colors = three_vertex_colors(beliefs, FACTOR_ANCHORS, alpha=1.5)

    # 2D triangle embedding
    coords_2d = simplex_to_triangle(beliefs)
    verts_2d = simplex_vertices_2d()

    fig = plt.figure(figsize=(12, 5))
    gs = GridSpec(1, 2, width_ratios=[1.5, 1.0], wspace=0.3)

    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax2d = fig.add_subplot(gs[0, 1])

    # 3D scatter
    ax3d.scatter(
        coords_3d[:, 0],
        coords_3d[:, 1],
        coords_3d[:, 2],
        s=6.0,
        c=colors,
        linewidths=0.0,
        depthshade=True,
    )
    style_3d_axes(ax3d, elev=30, azim=-60)
    set_equal_3d_limits(ax3d, coords_3d)
    ax3d.set_title(f"{title_prefix}\n3D PCA of synthetic joint", fontsize=13)

    # 2D simplex scatter
    ax2d.scatter(
        coords_2d[:, 0],
        coords_2d[:, 1],
        s=8.0,
        c=colors,
        alpha=0.7,
        linewidths=0.0,
    )
    # draw triangle edges
    tri = verts_2d
    for i in range(3):
        j = (i + 1) % 3
        x = [tri[i, 0], tri[j, 0]]
        y = [tri[i, 1], tri[j, 1]]
        ax2d.plot(x, y, color="black", linewidth=1.0)

    ax2d.set_aspect("equal")
    ax2d.set_xticks([])
    ax2d.set_yticks([])
    ax2d.set_title(f"{title_prefix}\n2D Simplex", fontsize=13)
    for spine in ax2d.spines.values():
        spine.set_visible(False)

    png_path = out_dir / "mess3_belief_geometry.png"
    pdf_path = out_dir / "mess3_belief_geometry.pdf"

    plt.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main():
    # 1) Build our mock 'mess3' HMM
    hmm = Mess3HMM(x=0.15, a=0.6, seed=7)

    # 2) Generate many beliefs (like generate_beliefs in mess3_beliefs.py)
    beliefs = generate_beliefs_batch(
        hmm,
        batch_size=400,   # can increase if you want more points
        seq_len=50,
        seed=7,
    )
    print("Beliefs shape:", beliefs.shape)  # (N, 3)

    # 3) Create output dir and plot
    out_dir = Path("mess3_geometry_outputs")
    make_belief_geometry_figure(beliefs, out_dir)




if __name__ == "__main__":
    main()