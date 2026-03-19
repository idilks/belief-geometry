"""
2x2 experiment: model size (2L vs 3L) x parameter regime (simple vs rich).

Runs 4 conditions sequentially, each: train -> analyze -> inspect -> gif.
For the 3L-simple condition, reuses existing checkpoints_long/ data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import shutil
import numpy as np

from scripts.train import train_with_snapshots
from scripts.analyze_geometry import main as analyze_main
from scripts.inspect_snapshots import main as inspect_main
from scripts.make_geometry_gif import main as gif_main
from src.transformer import Mess3Transformer


SIMPLE_PARAMS = [(0.8, 1.2), (0.4, 3.0), (0.6, 0.3)]
RICH_PARAMS = [(0.75, 1.5), (0.35, 2.0), (0.35, 0.25)]

CONDITIONS = {
    "2L_simple": {"n_layers": 2, "params": SIMPLE_PARAMS},
    "2L_rich":   {"n_layers": 2, "params": RICH_PARAMS},
    "3L_simple": {"n_layers": 3, "params": SIMPLE_PARAMS},  # reuse existing
    "3L_rich":   {"n_layers": 3, "params": RICH_PARAMS},
}

# regularized conditions: dropout + heavier weight decay + more data
REG_CONDITIONS = {
    "2L_simple_reg": {"n_layers": 2, "params": SIMPLE_PARAMS},
    "2L_rich_reg":   {"n_layers": 2, "params": RICH_PARAMS},
}


def reuse_3L_simple():
    """
    Reuse existing checkpoints_long/ for 3L-simple.
    Write a config.json if missing, identify best val-loss epoch.
    """
    snap_dir = Path("checkpoints_long")
    out_dir = Path("geometry_3L_simple")

    if not snap_dir.exists():
        print("checkpoints_long/ not found — will train 3L_simple from scratch")
        return False

    history_path = snap_dir / "history.npz"
    if not history_path.exists():
        print("no history.npz in checkpoints_long/ — will train from scratch")
        return False

    h = np.load(history_path)
    val_losses = h["val_loss"]
    best_epoch = int(np.argmin(val_losses)) + 1  # 1-indexed
    best_val_loss = float(val_losses[best_epoch - 1])

    # compute n_params from architecture
    tmp_model = Mess3Transformer(vocab_size=3, d_model=64, n_heads=4, n_layers=3, d_ff=128, max_len=15)
    n_params = sum(p.numel() for p in tmp_model.parameters())
    del tmp_model

    # write config.json if missing
    config_path = snap_dir / "config.json"
    if not config_path.exists():
        config = {
            "component_params": SIMPLE_PARAMS,
            "n_layers": 3,
            "d_model": 64,
            "n_heads": 4,
            "d_ff": 128,
            "n_params": n_params,
            "num_sequences_per_component": 3000,
            "seq_len": 16,
            "epochs": len(val_losses),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    # backfill n_params if missing from existing config
    with open(config_path) as f:
        config = json.load(f)
    if "n_params" not in config:
        config["n_params"] = n_params
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"3L_simple: reusing checkpoints_long/ (best_epoch={best_epoch}, val_loss={best_val_loss:.4f}, n_params={n_params:,})")
    print(f"{'='*60}")

    # run analysis
    analyze_main(snap_dir=str(snap_dir), out_dir=str(out_dir))
    inspect_main(snap_dir=str(snap_dir), out_dir=str(out_dir))
    gif_main(snap_dir=str(snap_dir), out_dir=str(out_dir))

    return True


def run_condition(name, n_layers, component_params, dropout=0.0,
                  weight_decay=0.01, num_sequences_per_component=3000,
                  epochs=150, patience=30):
    """Train one condition and run all analysis."""
    ckpt_dir = f"checkpoints_{name}"
    geo_dir = f"geometry_{name}"

    print(f"\n{'='*60}")
    print(f"CONDITION: {name} ({n_layers}L, params={component_params})")
    print(f"  dropout={dropout}, weight_decay={weight_decay}, n_seq={num_sequences_per_component}")
    print(f"{'='*60}")

    model, history, best_epoch = train_with_snapshots(
        component_params=component_params,
        n_layers=n_layers,
        dropout=dropout,
        weight_decay=weight_decay,
        num_sequences_per_component=num_sequences_per_component,
        epochs=epochs,
        patience=patience,
        save_dir=ckpt_dir,
    )

    # analysis
    analyze_main(snap_dir=ckpt_dir, out_dir=geo_dir)
    inspect_main(snap_dir=ckpt_dir, out_dir=geo_dir)
    gif_main(snap_dir=ckpt_dir, out_dir=geo_dir)

    # cleanup: keep only best-epoch snapshot + first/last + config/history
    snap_dir = Path(ckpt_dir)
    keep_epochs = {1, 5, best_epoch}
    snaps = list(snap_dir.glob("activations_epoch*.npz"))
    models = list(snap_dir.glob("model_epoch*.pt"))
    for f in snaps + models:
        epoch_num = int(f.stem.split("epoch")[1].split(".")[0])
        if epoch_num not in keep_epochs:
            f.unlink()

    print(f"  cleanup done: kept epochs {sorted(keep_epochs)}")


def main():
    # handle 3L_simple reuse first
    reused_3L = reuse_3L_simple()

    for name, cfg in CONDITIONS.items():
        if name == "3L_simple" and reused_3L:
            continue
        run_condition(name, cfg["n_layers"], cfg["params"])

    # regularized conditions
    for name, cfg in REG_CONDITIONS.items():
        run_condition(
            name, cfg["n_layers"], cfg["params"],
            dropout=0.15, weight_decay=0.1,
            num_sequences_per_component=10000,
            epochs=300, patience=50,
        )

    print(f"\n{'='*60}")
    print("ALL CONDITIONS COMPLETE")
    print(f"{'='*60}")
    print("\nrun scripts/compare_2x2.py to generate the comparison figure.")


if __name__ == "__main__":
    main()
