#!/usr/bin/env python3
"""Debug script to understand why optimization isn't working."""

import torch

from macrostat.estimation import mse_loss
from macrostat.models import get_model

# Generate target data
print("1. Generating target data with true parameters...")
model = get_model("GL06SIM")()
model.parameters["PropensityToConsumeIncome"] = 0.6
model.parameters["PropensityToConsumeSavings"] = 0.4

with torch.no_grad():
    target_output = model.simulate(scenario=0)

print(
    f"   Target ConsumptionDemand[50]: {target_output['ConsumptionDemand'][50].item():.4f}"
)
print()

# Perturb parameters
print("2. Perturbing parameters...")
model.parameters["PropensityToConsumeIncome"] = 0.7
model.parameters["PropensityToConsumeSavings"] = 0.3
print("   α1: 0.6 → 0.7")
print("   α2: 0.4 → 0.3")
print()

# Get behavior instance
print("3. Creating behavior instance...")
behavior = model.get_model_training_instance(scenario=0)

# Check parameters require grad
print("4. Checking parameter gradients...")
for name, p in behavior.named_parameters():
    val_str = f"{p.item():.4f}" if p.numel() == 1 else f"shape={p.shape}"
    print(f"   {name}: requires_grad={p.requires_grad}, {val_str}")
print()

# Check initial output
print("5. Computing initial output...")
with torch.no_grad():
    initial_output = behavior.forward()
print(
    f"   Initial ConsumptionDemand[50]: {initial_output['ConsumptionDemand'][50].item():.4f}"
)
print()

# Compute initial loss
print("6. Computing initial loss...")


def loss_fn(output):
    return mse_loss(
        output,
        target_output,
        variables=["ConsumptionDemand", "DisposableIncome"],
        timesteps=slice(-20, None),
        reduction="mean",
    )


with torch.no_grad():
    initial_loss = loss_fn(initial_output)
print(f"   Initial loss: {initial_loss.item():.6e}")
print()

# Test gradient computation
print("7. Testing gradient computation...")
behavior.zero_grad()
output = behavior.forward()
loss = loss_fn(output)
print(f"   Loss: {loss.item():.6e}")
loss.backward()

for name, p in behavior.named_parameters():
    if p.grad is None:
        grad_info = "None"
    elif p.numel() == 1:
        grad_info = f"{p.grad.item():.6e}"
    else:
        grad_info = f"shape={p.grad.shape}"
    print(f"   {name}: grad={grad_info}")
print()

# Test one optimizer step
print("8. Testing one optimizer step...")
params_to_optimize = [
    p for name, p in behavior.named_parameters() if name.startswith("params.")
]
optimizer = torch.optim.Adam(params_to_optimize, lr=1e-2)

print("   Before step:")
for name, p in behavior.named_parameters():
    val_str = f"{p.item():.6f}" if p.numel() == 1 else f"shape={p.shape}"
    print(f"      {name}: {val_str}")

optimizer.step()

print("   After step:")
for name, p in behavior.named_parameters():
    val_str = f"{p.item():.6f}" if p.numel() == 1 else f"shape={p.shape}"
    print(f"      {name}: {val_str}")
print()

# Test if loss changes
print("9. Testing if loss changes after step...")
with torch.no_grad():
    new_output = behavior.forward()
    new_loss = loss_fn(new_output)
print(f"   Old loss: {initial_loss.item():.6e}")
print(f"   New loss: {new_loss.item():.6e}")
print(f"   Change:   {(new_loss.item() - initial_loss.item()):.6e}")
