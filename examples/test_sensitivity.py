#!/usr/bin/env python3
"""Test if model output is sensitive to parameter changes."""

import torch

from macrostat.estimation import mse_loss
from macrostat.models import get_model

# Test with α1 = 0.6
model = get_model("GL06SIM")()
model.parameters["PropensityToConsumeIncome"] = 0.6
model.parameters["PropensityToConsumeSavings"] = 0.4

with torch.no_grad():
    output1 = model.simulate(scenario=0)

print("With α1=0.6, α2=0.4:")
print(f"  ConsumptionDemand[10]: {output1['ConsumptionDemand'][10].item():.4f}")
print(f"  ConsumptionDemand[50]: {output1['ConsumptionDemand'][50].item():.4f}")
print(f"  ConsumptionDemand[90]: {output1['ConsumptionDemand'][90].item():.4f}")
print()

# Test with α1 = 0.7
model.parameters["PropensityToConsumeIncome"] = 0.7
model.parameters["PropensityToConsumeSavings"] = 0.3

with torch.no_grad():
    output2 = model.simulate(scenario=0)

print("With α1=0.7, α2=0.3:")
print(f"  ConsumptionDemand[10]: {output2['ConsumptionDemand'][10].item():.4f}")
print(f"  ConsumptionDemand[50]: {output2['ConsumptionDemand'][50].item():.4f}")
print(f"  ConsumptionDemand[90]: {output2['ConsumptionDemand'][90].item():.4f}")
print()

print("Differences:")
print(
    f"  Timestep 10: {abs(output1['ConsumptionDemand'][10] - output2['ConsumptionDemand'][10]).item():.4f}"
)
print(
    f"  Timestep 50: {abs(output1['ConsumptionDemand'][50] - output2['ConsumptionDemand'][50]).item():.4f}"
)
print(
    f"  Timestep 90: {abs(output1['ConsumptionDemand'][90] - output2['ConsumptionDemand'][90]).item():.4f}"
)
print()


# Check if using early timesteps helps
def loss_early(out, tgt):
    return mse_loss(
        out,
        tgt,
        variables=["ConsumptionDemand"],
        timesteps=slice(5, 30),
        reduction="mean",
    )


def loss_late(out, tgt):
    return mse_loss(
        out,
        tgt,
        variables=["ConsumptionDemand"],
        timesteps=slice(-20, None),
        reduction="mean",
    )


loss_early_val = loss_early(output2, output1)
loss_late_val = loss_late(output2, output1)

print(f"MSE loss (timesteps 5-30): {loss_early_val.item():.6e}")
print(f"MSE loss (timesteps -20 to end): {loss_late_val.item():.6e}")
print()
print(
    f"Early timesteps have {loss_early_val.item() / loss_late_val.item():.1f}x more signal!"
)
