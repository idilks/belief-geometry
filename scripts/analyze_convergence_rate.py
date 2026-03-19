"""
Convergence rate analysis: how quickly does the transformer identify
which HMM component generated a sequence, compared to a Bayes classifier
that knows the true component HMMs?

What this script computes at each position t:
    1) Empirical accuracy:
             - Train a logistic regression probe on final-layer activations.
             - Evaluate held-out component classification accuracy.

    2) Bayesian accuracy (exact finite-T classifier in this script):
             - For each candidate component k, compute log P(y_0:t | C=k)
                 from the HMM forward recursion.
             - Predict argmax_k log P(y_0:t | C=k) (equal priors).
             - Compare with true component ID.

Important distinction:
    - This script computes Bayes accuracy exactly from finite prefixes.
    - The common curve exp(-t*h) is an asymptotic approximation for error decay
        rate, where h is a KL rate. It is not directly used for classification
        here.

Outputs per-condition plots and a combined comparison figure.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from src.hmm import Mess3HMM
from src.math_utils import symmetric_kl_rate
from src.dataset import DEFAULT_COMPONENTS


# ---------------------------------------------------------------------------
# Bayesian optimal classification
# ---------------------------------------------------------------------------

def bayesian_accuracy_curve(tokens, comp_ids, component_params):
    """
        Compute Bayes classification accuracy at each prefix length.

        Step-by-step math for one sequence y_0:t:

            Bayes rule:
                P(C=k | y_0:t) propto P(y_0:t | C=k) P(C=k)

            With equal priors P(C=k), maximizing the posterior is the same as
            maximizing the likelihood P(y_0:t | C=k).

            So the Bayes classifier is:
                k_hat_t = argmax_k log P(y_0:t | C=k)

        How log P(y_0:t | C=k) is computed:
            - Maintain filtered state alpha_t over hidden states for component k.
            - Update alpha_{t+1} = alpha_t @ T^(y_t).
            - Let Z_t = sum(alpha_{t+1}) before renormalization.
            - Then log P(y_0:t | C=k) = sum_{u=0}^t log Z_u.

        Why exponential terms appear in theory:
            If S_t = log(P(y_0:t|0)/P(y_0:t|1)), then converting from log-space to
            probability-space uses exp(), so asymptotic error laws look like
            exp(-t*h). Here h is a KL rate. That is a theory curve; this function
            computes exact finite-prefix Bayes decisions.
    """
    hmms = [Mess3HMM(s=s, r=r) for s, r in component_params]
    n_comp = len(hmms)
    N, T = tokens.shape

    # log_liks[i, k, t] stores log P(y_0:t | component k)
    # shape: (N, n_comp, T)
    log_liks = np.zeros((N, n_comp, T))

    for k, hmm in enumerate(hmms):
        for i in range(N):
            alpha = hmm.pi.copy()
            cum_ll = 0.0
            for t in range(T):
                # Unnormalized forward step for observed token y_t.
                alpha = alpha @ hmm.T_x[tokens[i, t]]
                norm = alpha.sum()
                if norm > 0:
                    # Add log normalizer: this is the per-step predictive
                    # probability contribution to the sequence log-likelihood.
                    cum_ll += np.log(norm)
                    alpha /= norm
                log_liks[i, k, t] = cum_ll

    # Equal-prior Bayes decision at each t: argmax_k log P(y_0:t | C=k)
    predictions = np.argmax(log_liks, axis=1)  # (N, T)
    true_labels = comp_ids[:, None]  # (N, 1)
    correct = (predictions == true_labels)  # (N, T)
    accuracy = correct.mean(axis=0)  # (T,)

    return accuracy


# ---------------------------------------------------------------------------
# Empirical (transformer) classification
# ---------------------------------------------------------------------------

def empirical_accuracy_curve(activations, comp_ids):
    """
    At each position t, train a logistic regression on final-layer activations
    and report test accuracy. 80/20 split, fixed seed.
    """
    N, T, d = activations.shape
    accuracies = np.zeros(T)

    idx = np.random.RandomState(0).permutation(N)
    split = int(0.8 * N)
    train_idx, test_idx = idx[:split], idx[split:]

    y_train = comp_ids[train_idx]
    y_test = comp_ids[test_idx]

    for t in range(T):
        X_train = activations[train_idx, t, :]
        X_test = activations[test_idx, t, :]

        clf = LogisticRegression(max_iter=1000, random_state=0)
        clf.fit(X_train, y_train)
        accuracies[t] = clf.score(X_test, y_test)

    return accuracies


# ---------------------------------------------------------------------------
# KL rate table
# ---------------------------------------------------------------------------

def pairwise_kl_table(component_params):
    """Compute pairwise symmetric KL rates and T* values."""
    hmms = [Mess3HMM(s=s, r=r) for s, r in component_params]
    n = len(hmms)
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            h = symmetric_kl_rate(hmms[i], hmms[j])
            rows.append({
                "pair": f"comp {i} vs {j}",
                "params_i": component_params[i],
                "params_j": component_params[j],
                "h_sym": h,
                "T_star": 1.0 / h if h > 0 else float("inf"),
            })
    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_convergence(empirical_acc, bayesian_acc, kl_table, condition_name, out_path):
    """Single-condition convergence plot."""
    T = len(empirical_acc)
    positions = np.arange(T)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(positions, bayesian_acc, "k-o", markersize=4, linewidth=2,
            label="Bayesian optimal", zorder=3)
    ax.plot(positions, empirical_acc, "s-", color="#2166ac", markersize=5,
            linewidth=2, label="Transformer (logistic probe)", zorder=3)

    # annotate T* for each pair
    h_min = min(row["h_sym"] for row in kl_table)
    T_star = 1.0 / h_min if h_min > 0 else T
    if T_star < T:
        ax.axvline(T_star, color="gray", ls="--", lw=1, alpha=0.7)
        ax.text(T_star + 0.2, 0.4, f"T* = {T_star:.1f}\n(hardest pair)",
                fontsize=8, color="gray")

    ax.axhline(1/3, color="lightgray", ls=":", lw=1, label="chance (1/3)")
    ax.set_xlabel("sequence position t")
    ax.set_ylabel("3-way classification accuracy")
    ax.set_title(f"Component identification convergence — {condition_name}")
    ax.set_ylim(0.25, 1.05)
    ax.set_xlim(-0.5, T - 0.5)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def plot_combined(results, out_path):
    """All conditions on one figure, 2x1: empirical curves + gap from Bayesian."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#2166ac", "#b2182b", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]
    markers = ["o", "s", "D", "^", "v", "<"]

    ax1, ax2 = axes

    for i, (name, emp, bay, _kl) in enumerate(results):
        T = len(emp)
        positions = np.arange(T)
        c = colors[i % len(colors)]
        m = markers[i % len(markers)]

        ax1.plot(positions, emp, f"-{m}", color=c, markersize=4, linewidth=1.5,
                 label=f"{name} (transformer)")
        ax1.plot(positions, bay, "--", color=c, linewidth=1, alpha=0.5)

        gap = bay - emp
        ax2.plot(positions, gap, f"-{m}", color=c, markersize=4, linewidth=1.5,
                 label=name)

    ax1.axhline(1/3, color="lightgray", ls=":", lw=1)
    ax1.set_xlabel("sequence position t")
    ax1.set_ylabel("classification accuracy")
    ax1.set_title("Convergence curves (dashed = Bayesian optimal)")
    ax1.set_ylim(0.25, 1.05)
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(True, alpha=0.3)

    ax2.axhline(0, color="lightgray", ls=":", lw=1)
    ax2.set_xlabel("sequence position t")
    ax2.set_ylabel("accuracy gap (Bayesian - transformer)")
    ax2.set_title("Optimality gap")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_config(checkpoint_dir):
    config_path = Path(checkpoint_dir) / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return None


def get_component_params(checkpoint_dir):
    config = load_config(checkpoint_dir)
    if config and "component_params" in config:
        return [tuple(p) for p in config["component_params"]]
    return DEFAULT_COMPONENTS


def find_best_activation_file(checkpoint_dir):
    """Find the activation file closest to the best epoch."""
    checkpoint_dir = Path(checkpoint_dir)
    config = load_config(checkpoint_dir)
    best_epoch = config.get("best_epoch", None) if config else None

    snap_files = sorted(checkpoint_dir.glob("activations_epoch*.npz"))
    if not snap_files:
        return None

    if best_epoch is not None:
        # find closest epoch
        epochs = []
        for f in snap_files:
            epoch_str = f.stem.replace("activations_epoch", "")
            epochs.append((int(epoch_str), f))
        # prefer exact match, otherwise closest
        for ep, f in epochs:
            if ep == best_epoch:
                return f
        # closest
        epochs.sort(key=lambda x: abs(x[0] - best_epoch))
        return epochs[0][1]

    # fallback: last snapshot
    return snap_files[-1]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyze_one(checkpoint_dir, out_dir, condition_name=None):
    """Run convergence analysis for one condition."""
    checkpoint_dir = Path(checkpoint_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if condition_name is None:
        condition_name = checkpoint_dir.name

    component_params = get_component_params(checkpoint_dir)
    print(f"\n{'='*60}")
    print(f"condition: {condition_name}")
    print(f"component params: {component_params}")

    # load activations
    snap_file = find_best_activation_file(checkpoint_dir)
    if snap_file is None:
        print(f"  no activation files found in {checkpoint_dir}, skipping")
        return None
    print(f"using activations: {snap_file.name}")
    data = np.load(snap_file)
    tokens = data["tokens"]       # (N, T)
    comp_ids = data["comp_ids"]   # (N,)
    final_acts = data["final"]    # (N, T, d)
    N, T, d = final_acts.shape
    print(f"  N={N}, T={T}, d={d}")

    # pairwise KL rates
    kl_table = pairwise_kl_table(component_params)
    print("\npairwise symmetric KL rates:")
    for row in kl_table:
        print(f"  {row['pair']}: h={row['h_sym']:.4f}, T*={row['T_star']:.1f}")

    # Bayesian optimal curve
    print("\ncomputing Bayesian optimal accuracy...")
    bayesian_acc = bayesian_accuracy_curve(tokens, comp_ids, component_params)
    print(f"  Bayesian accuracy: {bayesian_acc[0]:.3f} -> {bayesian_acc[-1]:.3f}")

    # empirical curve
    print("computing empirical accuracy (logistic probe at each position)...")
    empirical_acc = empirical_accuracy_curve(final_acts, comp_ids)
    print(f"  empirical accuracy: {empirical_acc[0]:.3f} -> {empirical_acc[-1]:.3f}")

    # plot
    plot_convergence(
        empirical_acc, bayesian_acc, kl_table,
        condition_name,
        out_dir / "convergence_rate.png",
    )

    return condition_name, empirical_acc, bayesian_acc, kl_table


def main():
    parser = argparse.ArgumentParser(description="Convergence rate analysis")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="single checkpoint dir to analyze")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="output directory for plots")
    parser.add_argument("--all", action="store_true",
                        help="run all 4 conditions and produce comparison plot")
    args = parser.parse_args()

    if args.all or args.checkpoint_dir is None:
        # run all conditions
        conditions = [
            ("checkpoints_2L_simple", "geometry_2L_simple", "2L simple"),
            ("checkpoints_2L_rich", "geometry_2L_rich", "2L rich"),
            ("checkpoints_2L_simple_reg", "geometry_2L_simple_reg", "2L simple+reg"),
            ("checkpoints_2L_rich_reg", "geometry_2L_rich_reg", "2L rich+reg"),
        ]

        results = []
        for ckpt_dir, geom_dir, name in conditions:
            if not Path(ckpt_dir).exists():
                print(f"skipping {name}: {ckpt_dir} not found")
                continue
            result = analyze_one(ckpt_dir, geom_dir, name)
            if result is not None:
                results.append(result)

        if len(results) > 1:
            combined_dir = Path("geometry_2x2")
            combined_dir.mkdir(exist_ok=True)
            plot_combined(results, combined_dir / "convergence_rate_comparison.png")

        print(f"\ndone. analyzed {len(results)} conditions.")

    else:
        out_dir = args.out_dir or f"geometry_{Path(args.checkpoint_dir).name}"
        analyze_one(args.checkpoint_dir, out_dir)


if __name__ == "__main__":
    main()
