"""
Run just the regularized conditions: dropout=0.15, weight_decay=0.1, 10K seqs/component.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train import train_with_snapshots
from scripts.analyze_geometry import main as analyze_main
from scripts.inspect_snapshots import main as inspect_main
from scripts.make_geometry_gif import main as gif_main


SIMPLE_PARAMS = [(0.8, 1.2), (0.4, 3.0), (0.6, 0.3)]
RICH_PARAMS = [(0.75, 1.5), (0.35, 2.0), (0.35, 0.25)]

REG_CONDITIONS = {
    "2L_simple_reg": {"n_layers": 2, "params": SIMPLE_PARAMS},
    "2L_rich_reg":   {"n_layers": 2, "params": RICH_PARAMS},
}


def run_reg_condition(name, n_layers, component_params):
    ckpt_dir = f"checkpoints_{name}"
    geo_dir = f"geometry_{name}"

    print(f"\n{'='*60}")
    print(f"CONDITION: {name} ({n_layers}L, params={component_params})")
    print(f"  dropout=0.15, weight_decay=0.1, n_seq=10000, epochs=300, patience=50")
    print(f"{'='*60}")

    model, history, best_epoch = train_with_snapshots(
        component_params=component_params,
        n_layers=n_layers,
        dropout=0.15,
        weight_decay=0.1,
        num_sequences_per_component=10000,
        epochs=300,
        patience=50,
        save_dir=ckpt_dir,
        snapshot_epochs=[1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200, 250, 300],
    )

    analyze_main(snap_dir=ckpt_dir, out_dir=geo_dir)
    inspect_main(snap_dir=ckpt_dir, out_dir=geo_dir)
    gif_main(snap_dir=ckpt_dir, out_dir=geo_dir)

    print(f"  done: {name}, best_epoch={best_epoch}")


if __name__ == "__main__":
    for name, cfg in REG_CONDITIONS.items():
        run_reg_condition(name, cfg["n_layers"], cfg["params"])

    print(f"\n{'='*60}")
    print("REGULARIZED CONDITIONS COMPLETE")
    print(f"{'='*60}")
