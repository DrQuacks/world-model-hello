import torch

from src.model import DynamicsMLP, ModelConfig

model = DynamicsMLP(ModelConfig(input_dim=6, output_dim=4, hidden_dim=128, num_hidden_layers=2))

x = torch.randn(10, 6)  # batch of 10 examples
y = model(x)

print("x shape:", x.shape)
print("y shape:", y.shape)  # should be (10, 4)
