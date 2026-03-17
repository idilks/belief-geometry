"""
Non-ergodic Mess3 dataset for next-token prediction.

Each sequence is drawn entirely from one ergodic component (one (x, a) pair).
The transformer never sees the component label — it's metadata for analysis.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from mess3_hmm_version1 import Mess3HMM


# --- Component definitions ---------------------------------------------------
# Each tuple is (x, a).
# x: off-diagonal asymmetry in transitions (0 = symmetric)
# a: switching rate (0 = always stay, 1 = always switch)
#
# These were chosen to occupy different dynamical regimes:
#   0: sticky, nearly symmetric  — beliefs hug one vertex for long stretches
#   1: volatile, strongly asymmetric — beliefs jump around the simplex unevenly
#   2: moderate switching, strong asymmetry — intermediate but with a directional bias
DEFAULT_COMPONENTS = [
    (0.05, 0.3),   # sticky, nearly symmetric
    (0.40, 0.7),   # volatile, strongly asymmetric
    (0.15, 0.9),   # fast-switching, moderate asymmetry
]


class Mess3NonErgodicDataset(Dataset):
    """
    Pre-generated dataset of observation sequences from a non-ergodic Mess3 mixture.

    Each sequence comes from exactly one ergodic component. Sequences are stored
    as int64 tensors with values in {0, 1, 2}.

    Returns (input_ids, targets, component_id) where:
        input_ids:    (seq_len - 1,)  observations 0..T-2
        targets:      (seq_len - 1,)  observations 1..T-1
        component_id: int             which component generated this sequence
    """

    def __init__(
        self,
        component_params=None,
        num_sequences_per_component: int = 1000,
        seq_len: int = 16,
        seed: int = 42,
    ):
        if component_params is None:
            component_params = DEFAULT_COMPONENTS

        self.component_params = component_params
        self.num_sequences_per_component = num_sequences_per_component
        self.seq_len = seq_len
        self.num_components = len(component_params)

        rng = np.random.default_rng(seed)

        # pre-generate everything
        self.observations = []   # list of (seq_len,) int arrays
        self.component_ids = []  # list of ints

        for comp_id, (x, a) in enumerate(component_params):
            hmm = Mess3HMM(x=x, a=a, seed=int(rng.integers(2**31)))
            for _ in range(num_sequences_per_component):
                hmm.rng = np.random.default_rng(int(rng.integers(2**31)))
                _z, y = hmm.sample_sequence(seq_len)
                self.observations.append(y)
                self.component_ids.append(comp_id)

        # stack into contiguous arrays
        self.observations = np.stack(self.observations)        # (N, seq_len)
        self.component_ids = np.array(self.component_ids)      # (N,)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        obs = self.observations[idx]
        input_ids = torch.tensor(obs[:-1], dtype=torch.long)
        targets = torch.tensor(obs[1:], dtype=torch.long)
        component_id = int(self.component_ids[idx])
        return input_ids, targets, component_id


def make_dataloaders(
    component_params=None,
    num_sequences_per_component: int = 1000,
    seq_len: int = 16,
    batch_size: int = 64,
    val_fraction: float = 0.1,
    seed: int = 42,
):
    """Build train/val DataLoaders with a random split."""
    dataset = Mess3NonErgodicDataset(
        component_params=component_params,
        num_sequences_per_component=num_sequences_per_component,
        seq_len=seq_len,
        seed=seed,
    )

    n = len(dataset)
    n_val = int(n * val_fraction)
    n_train = n - n_val

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=generator
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, dataset


def generate_component_beliefs(component_params=None, num_sequences=500, seq_len=64, seed=7):
    """
    Generate beliefs for each component separately. For visualization/analysis.

    Returns:
        beliefs_by_component: list of (num_sequences * seq_len, 3) arrays
        component_params: the params used
    """
    if component_params is None:
        component_params = DEFAULT_COMPONENTS

    rng = np.random.default_rng(seed)
    beliefs_by_component = []

    for x, a in component_params:
        hmm = Mess3HMM(x=x, a=a, seed=int(rng.integers(2**31)))
        all_beliefs = []
        for _ in range(num_sequences):
            hmm.rng = np.random.default_rng(int(rng.integers(2**31)))
            _z, y = hmm.sample_sequence(seq_len)
            alpha = hmm.forward_beliefs(y)
            all_beliefs.append(alpha)
        all_beliefs = np.concatenate(all_beliefs, axis=0)  # (num_sequences * seq_len, 3)
        beliefs_by_component.append(all_beliefs)

    return beliefs_by_component, component_params


# --- Quick verification -------------------------------------------------------
if __name__ == "__main__":
    print("=== Dataset verification ===\n")

    ds = Mess3NonErgodicDataset(seq_len=16, num_sequences_per_component=100)
    print(f"Dataset size: {len(ds)}")
    print(f"Components: {ds.num_components}, sequences/component: {ds.num_sequences_per_component}")

    input_ids, targets, comp_id = ds[0]
    print(f"\nds[0] shapes: input_ids={input_ids.shape}, targets={targets.shape}, comp_id={comp_id}")
    print(f"  input_ids: {input_ids.tolist()}")
    print(f"  targets:   {targets.tolist()}")
    print(f"  (targets should be input_ids shifted by 1)")

    # token frequencies per component
    print("\nToken frequencies per component:")
    for c in range(ds.num_components):
        mask = ds.component_ids == c
        obs = ds.observations[mask]
        counts = np.bincount(obs.ravel(), minlength=3)
        freqs = counts / counts.sum()
        x, a = ds.component_params[c]
        print(f"  Component {c} (x={x}, a={a}): {freqs.round(3)}")

    # dataloader check
    train_loader, val_loader, _ = make_dataloaders(
        num_sequences_per_component=100, batch_size=32
    )
    batch = next(iter(train_loader))
    inp, tgt, cids = batch
    print(f"\nDataLoader batch: input={inp.shape}, target={tgt.shape}")
    print(f"  Component IDs in batch: {sorted(cids.tolist())}")
    unique, counts = np.unique(cids.numpy(), return_counts=True)
    print(f"  Unique components: {dict(zip(unique, counts))}  (should be mixed if shuffle works)")
