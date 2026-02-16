import torch

from src.dataset import CartPoleTransitionsDataset

ds = CartPoleTransitionsDataset("data/cartpole_transitions.npz", target_mode="delta")

print("Dataset length:", len(ds))
print("Stats:", ds.stats)

x0, y0 = ds[0]
print("x0 shape:", x0.shape)  # should be torch.Size([6])
print("y0 shape:", y0.shape)  # should be torch.Size([4])

# Show a couple rows
for i in range(3):
    x, y = ds[i]
    print(f"\n{i}:")
    print("x =", x)
    print("y =", y)