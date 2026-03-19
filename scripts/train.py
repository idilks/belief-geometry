"""
Training with periodic snapshots for geometry evolution analysis.
Saves activations at regular intervals for GIF generation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
import numpy as np
import torch
import torch.nn as nn

from src.dataset import make_dataloaders, Mess3NonErgodicDataset
from src.transformer import Mess3Transformer


def train_with_snapshots(
    # data
    component_params=None,
    num_sequences_per_component=3000,
    seq_len=16,
    batch_size=64,
    # model
    d_model=64,
    n_heads=4,
    n_layers=3,
    d_ff=128,
    dropout=0.0,
    # training
    lr=3e-4,
    weight_decay=0.01,
    epochs=150,
    patience=30,
    seed=42,
    # snapshots
    snapshot_epochs=None,
    save_dir="checkpoints_long",
):
    if snapshot_epochs is None:
        snapshot_epochs = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150]

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")
    context_len = seq_len - 1

    # --- data ---
    print(f"building dataset: {num_sequences_per_component} seqs/component, seq_len={seq_len}")
    train_loader, val_loader, dataset = make_dataloaders(
        component_params=component_params,
        num_sequences_per_component=num_sequences_per_component,
        seq_len=seq_len,
        batch_size=batch_size,
        seed=seed,
    )
    print(f"train batches: {len(train_loader)}, val batches: {len(val_loader)}")

    # --- model ---
    model = Mess3Transformer(
        vocab_size=3,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_len=context_len,
        dropout=dropout,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_layers} layers, d={d_model}, heads={n_heads}, params={n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    # save config
    config = {
        "component_params": list(dataset.component_params),
        "n_layers": n_layers,
        "d_model": d_model,
        "n_heads": n_heads,
        "d_ff": d_ff,
        "n_params": n_params,
        "num_sequences_per_component": num_sequences_per_component,
        "seq_len": seq_len,
        "dropout": dropout,
        "epochs": epochs,
        "patience": patience,
        "lr": lr,
        "weight_decay": weight_decay,
        "seed": seed,
    }
    with open(save_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"baseline loss (uniform): {np.log(3):.4f}")
    print(f"snapshots at epochs: {snapshot_epochs}\n")

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # train
        model.train()
        total_loss = 0.0
        n_batches = 0
        for input_ids, targets, _comp_ids in train_loader:
            logits = model(input_ids)
            loss = criterion(logits.reshape(-1, 3), targets.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        train_loss = total_loss / n_batches
        history["train_loss"].append(train_loss)

        # val
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for input_ids, targets, _comp_ids in val_loader:
                logits = model(input_ids)
                loss = criterion(logits.reshape(-1, 3), targets.reshape(-1))
                val_loss += loss.item()
                preds = logits.argmax(dim=-1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.numel()

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # early stopping tracking
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), save_path / "model_best.pt")
            # snapshot at best epoch so compare_2x2 always has an exact match
            snap_file = save_path / f"activations_epoch{epoch:03d}.npz"
            if not snap_file.exists():
                extract_activations_snapshot(model, dataset, snap_file)
        else:
            epochs_without_improvement += 1

        scheduler.step()
        elapsed = time.time() - t0

        if epoch <= 5 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"epoch {epoch:3d}/{epochs} | "
                f"train {train_loss:.4f} | val {val_loss:.4f} | "
                f"acc {val_acc:.3f} | {elapsed:.1f}s"
            )

        # snapshot
        if epoch in snapshot_epochs:
            snap_file = save_path / f"activations_epoch{epoch:03d}.npz"
            extract_activations_snapshot(model, dataset, snap_file)
            torch.save(model.state_dict(), save_path / f"model_epoch{epoch:03d}.pt")
            print(f"  -> snapshot saved at epoch {epoch}")

        # early stopping (decoupled from snapshot schedule)
        if epochs_without_improvement >= patience:
            print(f"  early stopping at epoch {epoch} (best={best_epoch}, val_loss={best_val_loss:.4f})")
            break

    # update config with results
    config["best_epoch"] = best_epoch
    config["best_val_loss"] = float(best_val_loss)
    config["final_epoch"] = epoch
    with open(save_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # final save
    torch.save(model.state_dict(), save_path / "model_final.pt")
    np.savez(
        save_path / "history.npz",
        train_loss=history["train_loss"],
        val_loss=history["val_loss"],
        val_acc=history["val_acc"],
    )
    print(f"\ntraining complete. best epoch={best_epoch}, best val_loss={best_val_loss:.4f}")
    print(f"checkpoints in {save_path}/")
    return model, history, best_epoch


def extract_activations_snapshot(model, dataset, save_file, n_samples=1000):
    """Extract activations for a fixed subsample. Saves only final + embed layers."""
    model.eval()

    # deterministic subsample: first n_samples/3 from each component
    per_comp = n_samples // dataset.num_components
    indices = []
    comp_ids_arr = dataset.component_ids
    for c in range(dataset.num_components):
        comp_indices = np.where(comp_ids_arr == c)[0][:per_comp]
        indices.append(comp_indices)
    indices = np.concatenate(indices)

    subset = torch.utils.data.Subset(dataset, indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=512, shuffle=False)

    all_embed = []
    all_final = []
    all_comp_ids = []
    all_tokens = []

    with torch.no_grad():
        for input_ids, _targets, comp_ids in loader:
            residuals = model.extract_residual_stream(input_ids)
            all_embed.append(residuals["embed"].numpy())
            all_final.append(residuals["final"].numpy())
            all_comp_ids.append(comp_ids.numpy() if isinstance(comp_ids, torch.Tensor) else np.array(comp_ids))
            all_tokens.append(input_ids.numpy())

    np.savez(
        save_file,
        comp_ids=np.concatenate(all_comp_ids),
        tokens=np.concatenate(all_tokens),
        embed=np.concatenate(all_embed),
        final=np.concatenate(all_final),
    )


if __name__ == "__main__":
    train_with_snapshots()
