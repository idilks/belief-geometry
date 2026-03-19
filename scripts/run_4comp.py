"""
Train and analyze a 4-component model.

Tests whether K=4 components (vs K=3) forces head specialization
in a 2L/4H/d=64 transformer, or whether the distributed/redundant
computation story holds.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train import train_with_snapshots
from scripts.analyze_geometry import main as analyze_geometry
from scripts.analyze_mechanism import main as analyze_mechanism

# 4 components: the original 3 + a symmetric moderate one
FOUR_COMPONENTS = [
    (0.8, 1.2),   # sticky, mild clockwise
    (0.4, 3.0),   # volatile, strong clockwise
    (0.6, 0.3),   # moderate, counterclockwise
    (0.5, 1.0),   # moderate, symmetric (new)
]

CKPT_DIR = "checkpoints_4comp"
GEO_DIR = "geometry_4comp"
MECH_DIR = "mechanism_4comp"


def main():
    print("=== 4-component experiment ===")
    print(f"components: {FOUR_COMPONENTS}")

    # train with regularization (same setup as 2L_simple_reg)
    model, history, best_epoch = train_with_snapshots(
        component_params=FOUR_COMPONENTS,
        n_layers=2,
        d_model=64,
        n_heads=4,
        d_ff=128,
        dropout=0.15,
        weight_decay=0.1,
        num_sequences_per_component=10000,
        seq_len=16,
        epochs=300,
        patience=50,
        save_dir=CKPT_DIR,
    )

    print(f"\n--- geometry analysis ---")
    analyze_geometry(snap_dir=CKPT_DIR, out_dir=GEO_DIR)

    print(f"\n--- mechanism analysis ---")
    analyze_mechanism(checkpoint_dir=CKPT_DIR, out_dir=MECH_DIR)

    print(f"\ndone. checkpoints in {CKPT_DIR}/, geometry in {GEO_DIR}/, mechanism in {MECH_DIR}/")


if __name__ == "__main__":
    main()
