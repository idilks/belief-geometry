"""
Diagram of the experiment setup: data generation → training → analysis.
Shows how each component emits the same tokens {0,1,2} with different statistics.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from src.hmm import Mess3HMM


def make_diagram():
    fig = plt.figure(figsize=(16, 11))

    # ===================================================================
    # TOP SECTION: The three components and their emission statistics
    # ===================================================================
    # Show emission matrices for each component
    components = [
        (0.8, 1.2, "Component 0\n(sticky, mild CW)"),
        (0.4, 3.0, "Component 1\n(volatile, strong CW)"),
        (0.6, 0.3, "Component 2\n(moderate, CCW)"),
    ]

    comp_colors = ["#E31A1C", "#FF8C00", "#1F78B4"]

    # --- Row 1: Component boxes with emission probabilities ---
    ax_top = fig.add_axes([0.02, 0.62, 0.96, 0.35])
    ax_top.set_xlim(0, 10)
    ax_top.set_ylim(0, 4)
    ax_top.axis("off")

    # Title
    ax_top.text(5, 3.8, "DATA GENERATION", fontsize=16, fontweight="bold",
                ha="center", va="top", family="monospace")
    ax_top.text(5, 3.45, "each sequence: pick one component, generate entire sequence from it",
                fontsize=10, ha="center", va="top", color="gray")

    box_width = 2.8
    box_height = 2.6
    gap = 0.3
    total_width = 3 * box_width + 2 * gap
    x_start = (10 - total_width) / 2

    for i, (s, r, label) in enumerate(components):
        hmm = Mess3HMM(s=s, r=r)
        x = x_start + i * (box_width + gap)
        y = 0.4

        # Component box
        rect = FancyBboxPatch((x, y), box_width, box_height,
                              boxstyle="round,pad=0.1",
                              facecolor=comp_colors[i], alpha=0.12,
                              edgecolor=comp_colors[i], linewidth=2)
        ax_top.add_patch(rect)

        # Label
        ax_top.text(x + box_width/2, y + box_height - 0.15, label,
                    fontsize=10, fontweight="bold", ha="center", va="top",
                    color=comp_colors[i])

        # Emission table: P(token | state)
        ax_top.text(x + box_width/2, y + box_height - 0.65,
                    f"s={s}, r={r}", fontsize=9, ha="center", va="top",
                    family="monospace", color="#444")

        # Header
        ax_top.text(x + 0.3, y + box_height - 1.0, "P(token|state)",
                    fontsize=8, ha="left", va="top", style="italic", color="#666")

        col_labels = ["tok 0", "tok 1", "tok 2"]
        row_labels = ["state 0", "state 1", "state 2"]

        table_top = y + box_height - 1.25
        row_h = 0.35
        col_w = 0.65

        # Column headers
        for j, cl in enumerate(col_labels):
            ax_top.text(x + 1.0 + j * col_w, table_top, cl,
                        fontsize=7, ha="center", va="top", fontweight="bold",
                        family="monospace")

        # Rows
        for si in range(3):
            yt = table_top - (si + 1) * row_h
            ax_top.text(x + 0.25, yt, row_labels[si],
                        fontsize=7, ha="left", va="center", family="monospace")
            for tok in range(3):
                p = hmm.E[si, tok]  # P(emit tok | state si)
                # Bold the dominant emission
                weight = "bold" if p > 0.3 else "normal"
                color = "black" if p > 0.3 else "#888"
                ax_top.text(x + 1.0 + tok * col_w, yt, f"{p:.2f}",
                            fontsize=8, ha="center", va="center",
                            family="monospace", fontweight=weight, color=color)

        # Sample sequence
        hmm_sample = Mess3HMM(s=s, r=r, seed=42+i)
        _, y_seq = hmm_sample.sample_sequence(12)
        seq_str = "".join(str(t) for t in y_seq[:12])
        ax_top.text(x + box_width/2, y + 0.25,
                    f"e.g. {seq_str}...",
                    fontsize=8, ha="center", va="center",
                    family="monospace", color=comp_colors[i],
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor=comp_colors[i], alpha=0.8))

    # ===================================================================
    # MIDDLE SECTION: What the transformer sees
    # ===================================================================
    ax_mid = fig.add_axes([0.02, 0.38, 0.96, 0.22])
    ax_mid.set_xlim(0, 10)
    ax_mid.set_ylim(0, 3)
    ax_mid.axis("off")

    # Arrow from top to middle
    ax_mid.annotate("", xy=(5, 2.9), xytext=(5, 3.1),
                    arrowprops=dict(arrowstyle="-|>", color="gray", lw=2))

    # The "shuffle" box
    shuffle_rect = FancyBboxPatch((1.5, 1.5), 7, 1.2,
                                   boxstyle="round,pad=0.15",
                                   facecolor="#f0f0f0", edgecolor="#333",
                                   linewidth=1.5)
    ax_mid.add_patch(shuffle_rect)

    ax_mid.text(5, 2.45, "WHAT THE TRANSFORMER SEES", fontsize=12,
                fontweight="bold", ha="center", va="top", family="monospace")

    # Show shuffled sequences with color coding (but model doesn't see colors)
    np.random.seed(7)
    seqs = []
    for i, (s, r, _) in enumerate(components):
        hmm = Mess3HMM(s=s, r=r, seed=100+i)
        for j in range(3):
            hmm.rng = np.random.default_rng(200+i*10+j)
            _, y = hmm.sample_sequence(10)
            seqs.append((y, i))

    np.random.shuffle(seqs)

    x_pos = 1.8
    for idx, (seq, comp) in enumerate(seqs[:7]):
        seq_str = "".join(str(t) for t in seq[:8]) + "..."
        ax_mid.text(x_pos, 1.75, seq_str, fontsize=8, family="monospace",
                    ha="left", va="center", color="#333")
        x_pos += 0.95

    ax_mid.text(5, 1.15, "just sequences of {0, 1, 2}. no labels. no component IDs. all shuffled together.",
                fontsize=9, ha="center", va="top", color="#666", style="italic")

    # Key insight box
    insight_rect = FancyBboxPatch((1.0, 0.1), 8, 0.85,
                                   boxstyle="round,pad=0.1",
                                   facecolor="#fffde7", edgecolor="#f9a825",
                                   linewidth=1.5)
    ax_mid.add_patch(insight_rect)

    ax_mid.text(5, 0.65, "KEY: all 3 components emit the same tokens {0, 1, 2}",
                fontsize=11, fontweight="bold", ha="center", va="center")
    ax_mid.text(5, 0.3,
                "the components differ in HOW OFTEN each state emits each token (see tables above).\n"
                "the transformer must infer which component from token statistics alone.",
                fontsize=9, ha="center", va="center", color="#555")

    # ===================================================================
    # BOTTOM SECTION: Training objective + analysis
    # ===================================================================
    ax_bot = fig.add_axes([0.02, 0.02, 0.96, 0.34])
    ax_bot.set_xlim(0, 10)
    ax_bot.set_ylim(0, 4)
    ax_bot.axis("off")

    # Training box
    train_rect = FancyBboxPatch((0.5, 2.2), 4, 1.6,
                                 boxstyle="round,pad=0.15",
                                 facecolor="#e8f5e9", edgecolor="#388e3c",
                                 linewidth=1.5)
    ax_bot.add_patch(train_rect)

    ax_bot.text(2.5, 3.55, "TRAINING", fontsize=12, fontweight="bold",
                ha="center", va="top", family="monospace", color="#2e7d32")
    ax_bot.text(2.5, 3.15,
                "objective: predict next token\n"
                "input:  [0, 2, 2, 1, 0, ...]\n"
                "target: [2, 2, 1, 0, 2, ...]\n"
                "loss: cross-entropy",
                fontsize=9, ha="center", va="top", family="monospace", color="#333")

    # Analysis box
    analysis_rect = FancyBboxPatch((5.5, 2.2), 4, 1.6,
                                    boxstyle="round,pad=0.15",
                                    facecolor="#e3f2fd", edgecolor="#1565c0",
                                    linewidth=1.5)
    ax_bot.add_patch(analysis_rect)

    ax_bot.text(7.5, 3.55, "ANALYSIS (post-hoc)", fontsize=12, fontweight="bold",
                ha="center", va="top", family="monospace", color="#0d47a1")
    ax_bot.text(7.5, 3.15,
                "NOW we use comp_ids to:\n"
                "1. compute ground-truth beliefs\n"
                "   per component's (s,r)\n"
                "2. color-code activation plots\n"
                "3. measure R² per component",
                fontsize=9, ha="center", va="top", family="monospace", color="#333")

    # Arrow between them
    ax_bot.annotate("", xy=(5.4, 3.0), xytext=(4.6, 3.0),
                    arrowprops=dict(arrowstyle="-|>", color="gray", lw=2))

    # Bottom: the question
    q_rect = FancyBboxPatch((0.8, 0.1), 8.4, 1.8,
                             boxstyle="round,pad=0.15",
                             facecolor="#fce4ec", edgecolor="#c62828",
                             linewidth=1.5)
    ax_bot.add_patch(q_rect)

    ax_bot.text(5, 1.7, "THE QUESTION", fontsize=12, fontweight="bold",
                ha="center", va="top", family="monospace", color="#b71c1c")
    ax_bot.text(5, 1.3,
                "the transformer was only trained to predict next tokens.\n"
                "but did it internally learn the belief-state geometry of the underlying HMMs?\n\n"
                "test: can a linear map from 64D activations recover the 3D belief simplex?",
                fontsize=10, ha="center", va="top", color="#333")
    ax_bot.text(5, 0.3, "answer: R² ≈ 0.97  →  yes.",
                fontsize=12, fontweight="bold", ha="center", va="center",
                color="#2e7d32")

    fig.savefig("geometry_outputs/experiment_diagram.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("saved geometry_outputs/experiment_diagram.png")


if __name__ == "__main__":
    make_diagram()
