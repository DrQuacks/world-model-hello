import argparse
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.dataset import CartPoleTransitionsDataset
from src.model import DynamicsMLP, ModelConfig


def choose_device(device_str: str | None) -> torch.device:
    """
    Choose a device for training.
    - If device_str is provided ("cpu", "cuda", "mps"), use it.
    - Otherwise prefer CUDA, then MPS (Apple Silicon), then CPU.
    """
    if device_str is not None:
        return torch.device(device_str)

    if torch.cuda.is_available():
        return torch.device("cuda")

    # Apple Silicon GPU backend
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def make_splits(n: int, val_frac: float, seed: int) -> tuple[list[int], list[int]]:
    """
    Create a shuffled train/val split of indices [0..n-1].
    """
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()

    val_size = int(n * val_frac)
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]
    return train_idx, val_idx


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """
    Return average MSE loss over the loader.
    """
    model.eval()
    loss_fn = nn.MSELoss(reduction="sum")

    total_loss = 0.0
    total_examples = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)  # sum over batch
        total_loss += float(loss.item())
        total_examples += x.shape[0]

    return total_loss / total_examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/cartpole_transitions.npz", help="Path to .npz dataset")
    parser.add_argument("--target", type=str, choices=["delta", "next_obs"], default="delta", help="Prediction target")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None, help='e.g. "cpu", "cuda", "mps" (default: auto)')
    parser.add_argument("--save", type=str, default="checkpoints/dynamics_mlp.pt", help="Checkpoint path")
    args = parser.parse_args()

    device = choose_device(args.device)
    print("Using device:", device)

    # Reproducibility
    torch.manual_seed(args.seed)

    # Load dataset
    ds = CartPoleTransitionsDataset(args.data, target_mode=args.target)
    print("Dataset stats:", ds.stats)

    # Train/val split using indices
    train_idx, val_idx = make_splits(len(ds), val_frac=args.val_frac, seed=args.seed)
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, drop_last=False)

    # Build model dimensions from dataset stats
    input_dim = ds.stats.obs_dim + ds.stats.act_dim   # 4 + 2 = 6
    output_dim = ds.stats.obs_dim                     # 4

    config = ModelConfig(input_dim=input_dim, output_dim=output_dim, hidden_dim=128, num_hidden_layers=2, dropout=0.0)
    model = DynamicsMLP(config).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            num_batches += 1

        train_loss = running_loss / max(1, num_batches)
        val_loss = evaluate(model, val_loader, device=device)

        print(f"Epoch {epoch:02d} | train_mse={train_loss:.6f} | val_mse={val_loss:.6f}")

    # Save checkpoint
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "model_state_dict": model.state_dict(),
        "model_config": asdict(config),
        "dataset_stats": asdict(ds.stats),
        "target_mode": args.target,
    }
    torch.save(ckpt, save_path)
    print(f"\nSaved checkpoint to: {save_path}")


if __name__ == "__main__":
    main()
