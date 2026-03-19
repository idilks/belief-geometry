"""
Convergence analysis: what happens when two Mess3 components approach each other?

Mathematical argument:
  As (s2, r2) -> (s1, r1), the token-conditional matrices T2^x -> T1^x entry-wise.
    Therefore h(P1 || P2) -> 0 continuously. For long prefixes, Bayes error
    decays approximately like exp(-T * h), where h is a KL rate.

    Important: exp(-T*h) is a rate approximation, not an exact finite-T posterior
    probability for every setting. We use it as a principled trend/threshold rule:
    if T*h << 1, evidence is weak; if T*h >> 1, evidence is strong.

    In this interpretation, the critical scale is h* ~ 1/T.

  For our context length T=15, h* ~ 0.067.

This is testable: we interpolate between two components, compute h at each
point, and verify that subspace collapse occurs near the predicted threshold.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from src.hmm import Mess3HMM, generate_beliefs_batch
from src.dataset import DEFAULT_COMPONENTS, Mess3NonErgodicDataset, make_dataloaders
from src.belief_geometry import simplex_to_triangle, three_vertex_colors, get_comp_colors_hex
from src.math_utils import sequence_kl_rate, symmetric_kl_rate, emission_kl, cluster_separation, subspace_angles


CONTEXT_LEN = 15  # seq_len - 1


# ---------------------------------------------------------------------------
# Step 1: Theory-only analysis
# ---------------------------------------------------------------------------

def compute_kl_sweep(anchor_params, target_params, n_steps=20):
    """
    Interpolate (s, r) from target toward anchor. At each point compute:
    - parameter distance
    - emission KL
    - sequence KL rate
    - distinguishability horizon T* = 1/h
    """
    s0, r0 = anchor_params
    s1, r1 = target_params
    results = []

    for i in range(n_steps + 1):
        alpha = i / n_steps  # 0 = target, 1 = anchor
        s = s1 + alpha * (s0 - s1)
        r = r1 + alpha * (r0 - r1)

        hmm_anchor = Mess3HMM(s=s0, r=r0)
        hmm_interp = Mess3HMM(s=s, r=r)

        h_fwd = sequence_kl_rate(hmm_anchor, hmm_interp)
        h_rev = sequence_kl_rate(hmm_interp, hmm_anchor)
        h_sym = (h_fwd + h_rev) / 2
        e_kl = emission_kl(hmm_anchor, hmm_interp)
        param_dist = np.sqrt((s - s0)**2 + (r - r0)**2)

        results.append({
            "alpha": alpha,
            "s": s, "r": r,
            "param_dist": param_dist,
            "h_fwd": h_fwd, "h_rev": h_rev, "h_sym": h_sym,
            "emission_kl": e_kl,
            "T_star": 1.0 / max(h_sym, 1e-10),
        })

    return results


def plot_kl_sweep(sweep, out_dir, anchor_params, target_params):
    """Plot KL rate, emission KL, and T* vs interpolation parameter."""
    alphas = [r["alpha"] for r in sweep]
    h_syms = [r["h_sym"] for r in sweep]
    e_kls = [r["emission_kl"] for r in sweep]
    param_dists = [r["param_dist"] for r in sweep]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # left: KL rates vs parameter distance
    ax1.plot(param_dists, h_syms, "o-", color="#e41a1c", markersize=4, label="sequence KL rate h")
    ax1.plot(param_dists, e_kls, "s--", color="#1f78b4", markersize=4, label="emission KL")
    ax1.axhline(1.0 / CONTEXT_LEN, color="gray", ls=":", lw=1.5,
                label=f"h* = 1/T = {1.0/CONTEXT_LEN:.3f}")
    ax1.set_xlabel("parameter distance ||(s,r) - (s₀,r₀)||")
    ax1.set_ylabel("KL rate (nats/step)")
    ax1.set_title("KL divergence rate along interpolation path")
    ax1.legend()
    ax1.set_ylim(bottom=-0.01)

    # right: distinguishability horizon
    t_stars = [r["T_star"] for r in sweep]
    ax2.plot(param_dists, t_stars, "o-", color="#33a02c", markersize=4)
    ax2.axhline(CONTEXT_LEN, color="gray", ls=":", lw=1.5, label=f"T = {CONTEXT_LEN}")
    ax2.set_xlabel("parameter distance")
    ax2.set_ylabel("T* = 1/h (tokens needed)")
    ax2.set_title("Distinguishability horizon")
    ax2.set_ylim(0, min(max(t_stars) * 1.1, 500))
    ax2.legend()

    s0, r0 = anchor_params
    s1, r1 = target_params
    plt.suptitle(f"Interpolation: (s={s1}, r={r1}) → (s={s0}, r={r0})", fontsize=13)
    plt.tight_layout()
    out_path = out_dir / "kl_sweep.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def plot_kl_landscape(out_dir):
    """
    2D heatmap of h(anchor || (s, r)) over parameter space.
    Overlays default component locations and h* contour.
    """
    anchor = Mess3HMM(s=0.8, r=1.2)

    s_range = np.linspace(0.2, 0.95, 40)
    r_range = np.linspace(0.1, 5.0, 40)
    h_grid = np.zeros((len(r_range), len(s_range)))

    for i, r in enumerate(r_range):
        for j, s in enumerate(s_range):
            hmm = Mess3HMM(s=s, r=r)
            h_grid[i, j] = symmetric_kl_rate(anchor, hmm)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(s_range, r_range, h_grid, cmap="viridis", shading="auto")
    plt.colorbar(im, ax=ax, label="symmetric KL rate h")

    # contour at h* = 1/T
    cs = ax.contour(s_range, r_range, h_grid, levels=[1.0 / CONTEXT_LEN],
                    colors="white", linewidths=2, linestyles="--")
    ax.clabel(cs, fmt=f"h*=1/{CONTEXT_LEN}", fontsize=9, colors="white")

    # mark default components
    comp_colors = get_comp_colors_hex(len(DEFAULT_COMPONENTS))
    for idx, (s, r) in enumerate(DEFAULT_COMPONENTS):
        ax.plot(s, r, "o", color=comp_colors[idx], markersize=10, markeredgecolor="white", markeredgewidth=1.5)
        ax.annotate(f"  comp {idx}", (s, r), color="white", fontsize=9, fontweight="bold")

    ax.set_xlabel("s (self-transition)")
    ax.set_ylabel("r (asymmetry ratio)")
    ax.set_title("KL rate landscape: h(comp0 || (s, r))")
    plt.tight_layout()
    out_path = out_dir / "kl_landscape.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


# ---------------------------------------------------------------------------
# Step 2: Experimental verification (trains models)
# ---------------------------------------------------------------------------

def train_convergence_sweep(anchor_params, target_params, n_steps=8, out_dir=None):
    """
    For each interpolation point, train a 3-component model where comp 1
    is interpolated toward comp 0. Measure separation metrics.
    """
    import torch
    from src.transformer import Mess3Transformer
    from scripts.train import train_with_snapshots

    s0, r0 = anchor_params
    s1, r1 = target_params

    results = []

    for step in range(n_steps + 1):
        alpha = step / n_steps
        s_interp = s1 + alpha * (s0 - s1)
        r_interp = r1 + alpha * (r0 - r1)

        # 3 components: anchor, interpolated, and a third (fixed)
        comp_params = [
            anchor_params,
            (s_interp, r_interp),
            (0.6, 0.3),  # fixed third component
        ]

        h_sym = symmetric_kl_rate(
            Mess3HMM(s=s0, r=r0),
            Mess3HMM(s=s_interp, r=r_interp),
        )

        print(f"\n--- step {step}/{n_steps}: comp1=({s_interp:.3f}, {r_interp:.3f}), h={h_sym:.4f} ---")

        save_dir = out_dir / f"convergence_step{step:02d}" if out_dir else Path(f"convergence_step{step:02d}")

        t0 = time.time()
        model, history, best_epoch = train_with_snapshots(
            component_params=comp_params,
            num_sequences_per_component=1000,
            seq_len=16,
            batch_size=64,
            d_model=32,
            n_heads=4,
            n_layers=2,
            d_ff=64,
            dropout=0.0,
            lr=3e-4,
            weight_decay=0.01,
            epochs=50,
            patience=20,
            seed=42,
            snapshot_epochs=[50],
            save_dir=str(save_dir),
        )
        elapsed = time.time() - t0

        # extract activations with all layers
        dataset = Mess3NonErgodicDataset(
            component_params=comp_params,
            num_sequences_per_component=1000,
            seq_len=16,
            seed=42,
        )

        per_comp = 1000 // len(comp_params)
        indices = []
        for c in range(len(comp_params)):
            comp_indices = np.where(dataset.component_ids == c)[0][:per_comp]
            indices.append(comp_indices)
        indices = np.concatenate(indices)

        subset = torch.utils.data.Subset(dataset, indices)
        loader = torch.utils.data.DataLoader(subset, batch_size=512, shuffle=False)

        all_final = []
        all_comp_ids = []
        with torch.no_grad():
            for input_ids, _targets, comp_ids in loader:
                residuals = model.extract_residual_stream(input_ids)
                all_final.append(residuals["final"].numpy())
                all_comp_ids.append(comp_ids.numpy() if isinstance(comp_ids, torch.Tensor) else np.array(comp_ids))

        acts_final = np.concatenate(all_final)  # (N, T, d)
        comp_ids_arr = np.concatenate(all_comp_ids)

        # use last position for separation metrics
        last_pos = acts_final[:, -1, :]  # (N, d)

        # binary separation: comp 0 vs comp 1
        mask_01 = (comp_ids_arr == 0) | (comp_ids_arr == 1)
        acts_01 = last_pos[mask_01]
        labels_01 = comp_ids_arr[mask_01]

        sep = cluster_separation(acts_01, labels_01)
        angles = subspace_angles(
            acts_final[comp_ids_arr == 0].reshape(-1, acts_final.shape[-1]),
            acts_final[comp_ids_arr == 1].reshape(-1, acts_final.shape[-1]),
            n_components=3,
        )

        result = {
            "step": step,
            "alpha": alpha,
            "s": s_interp, "r": r_interp,
            "h_sym": h_sym,
            "silhouette": sep["silhouette"],
            "linear_acc": sep["linear_acc"],
            "principal_angles": angles.tolist(),
            "min_angle": float(angles.min()),
            "best_val_loss": history["val_loss"][best_epoch - 1] if best_epoch > 0 else history["val_loss"][-1],
            "elapsed": elapsed,
        }
        results.append(result)
        print(f"  silhouette={sep['silhouette']:.3f}, linear_acc={sep['linear_acc']:.3f}, "
              f"min_angle={np.degrees(angles.min()):.1f}°, time={elapsed:.0f}s")

    return results


def plot_phase_transition(sweep_theory, sweep_exp, out_dir):
    """
    x-axis: sequence KL rate h
    left y-axis: theoretical proxy P(error) ~= exp(-T*h)
    right y-axis: experimental separation metrics
    vertical line at h = 1/T

    Note: the left curve is an asymptotic proxy for error decay rate, used to
    visualize scaling with h. It is not an exact calibrated finite-sample error.
    """
    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax2 = ax1.twinx()

    # theoretical curve (continuous)
    h_range = np.linspace(0, max(r["h_sym"] for r in sweep_theory) * 1.1, 200)
    p_error = np.exp(-CONTEXT_LEN * h_range)
    ax1.plot(h_range, p_error, "-", color="#999999", lw=2, label="P(error) = exp(-T·h)")

    # experimental points
    h_exp = [r["h_sym"] for r in sweep_exp]
    sil_exp = [r["silhouette"] for r in sweep_exp]
    acc_exp = [r["linear_acc"] for r in sweep_exp]
    min_angle_exp = [np.degrees(r["min_angle"]) for r in sweep_exp]

    ax2.plot(h_exp, sil_exp, "o-", color="#e41a1c", markersize=6, label="silhouette score")
    ax2.plot(h_exp, acc_exp, "s-", color="#1f78b4", markersize=6, label="linear classifier acc")

    # threshold
    h_star = 1.0 / CONTEXT_LEN
    ax1.axvline(h_star, color="black", ls=":", lw=1.5, label=f"h* = 1/T = {h_star:.3f}")

    ax1.set_xlabel("sequence KL rate h (nats/step)")
    ax1.set_ylabel("P(error) = exp(-T·h)", color="#999999")
    ax2.set_ylabel("separation metric", color="#e41a1c")
    ax1.set_title("Phase transition: subspace collapse near h* = 1/T")

    # combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=9)

    ax1.set_xlim(left=-0.01)
    ax1.set_ylim(0, 1.05)
    ax2.set_ylim(-0.1, 1.05)

    plt.tight_layout()
    out_path = out_dir / "phase_transition.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


# ---------------------------------------------------------------------------
# Step 3: Belief fractal convergence visualization
# ---------------------------------------------------------------------------

def plot_fractal_convergence(anchor_params, target_params, n_steps=6, out_dir=None):
    """
    1xN grid showing belief fractals as params interpolate.
    Annotated with h value at each point.
    """
    s0, r0 = anchor_params
    s1, r1 = target_params

    fig, axes = plt.subplots(1, n_steps + 1, figsize=(3 * (n_steps + 1), 3.5))

    for step in range(n_steps + 1):
        alpha = step / n_steps
        s = s1 + alpha * (s0 - s1)
        r = r1 + alpha * (r0 - r1)

        hmm = Mess3HMM(s=s, r=r)
        beliefs = generate_beliefs_batch(hmm, batch_size=300, seq_len=50, seed=7)

        h = symmetric_kl_rate(Mess3HMM(s=s0, r=r0), hmm)

        tri_coords = simplex_to_triangle(beliefs)
        colors = three_vertex_colors(beliefs, alpha=1.5)

        ax = axes[step]
        ax.scatter(tri_coords[:, 0], tri_coords[:, 1], s=0.5, c=colors, alpha=0.5)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"s={s:.2f}, r={r:.2f}\nh={h:.4f}", fontsize=9)

    plt.suptitle(f"Belief fractal convergence: ({s1},{r1}) → ({s0},{r0})", fontsize=12)
    plt.tight_layout()
    out_path = (out_dir or Path("geometry_outputs")) / "fractal_convergence.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(out_dir="geometry_outputs", do_training=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    anchor = (0.8, 1.2)   # comp 0
    target = (0.4, 3.0)   # comp 1

    # --- theory ---
    print("=== KL sweep ===")
    sweep = compute_kl_sweep(anchor, target, n_steps=20)
    for r in sweep[::5]:
        print(f"  alpha={r['alpha']:.2f}, (s={r['s']:.2f}, r={r['r']:.2f}), "
              f"h={r['h_sym']:.4f}, T*={r['T_star']:.1f}")

    plot_kl_sweep(sweep, out_dir, anchor, target)

    print("\n=== KL landscape ===")
    plot_kl_landscape(out_dir)

    print("\n=== fractal convergence ===")
    plot_fractal_convergence(anchor, target, n_steps=6, out_dir=out_dir)

    # --- experimental ---
    if do_training:
        print("\n=== training convergence sweep ===")
        exp_results = train_convergence_sweep(anchor, target, n_steps=8, out_dir=out_dir)

        # save raw results
        results_path = out_dir / "convergence_results.json"
        with open(results_path, "w") as f:
            json.dump(exp_results, f, indent=2)
        print(f"saved {results_path}")

        plot_phase_transition(sweep, exp_results, out_dir)
    else:
        print("\nskipping training sweep (use --train to enable)")

    print("\ndone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="geometry_outputs")
    parser.add_argument("--train", action="store_true", help="run training sweep (slow)")
    parser.add_argument("--no-training", action="store_true", help="theory only (default)")
    args = parser.parse_args()
    main(out_dir=args.out_dir, do_training=args.train)
