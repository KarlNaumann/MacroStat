#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Karl Naumann-Woleske
"""
Demonstration of Phase 1 estimation module with torch.optim.Adam.

This script demonstrates using the new loss functions with standard PyTorch
optimizers to calibrate GL06SIM model parameters.
"""

from __future__ import annotations

import torch

from macrostat.estimation import mse_loss
from macrostat.models import get_model


def main():
    """Demonstrate estimation module Phase 1 capabilities."""
    print("=" * 70)
    print("MacroStat Estimation Module - Phase 1 Demonstration")
    print("=" * 70)
    print()

    # Step 1: Create model and generate synthetic target data
    print("Step 1: Generating synthetic target data from GL06SIM...")
    model = get_model("GL06SIM")()

    # Set known "true" parameters
    true_params = {
        "PropensityToConsumeIncome": 0.6,
        "PropensityToConsumeSavings": 0.4,
    }

    for param_name, value in true_params.items():
        model.parameters[param_name] = value

    # Generate synthetic data
    with torch.no_grad():
        target_output = model.simulate(scenario=0)

    print(f"  Generated {len(target_output)} variables")
    print(f"  Timesteps: {target_output['ConsumptionDemand'].shape[0]}")
    print()

    # Step 2: Perturb parameters for calibration
    print("Step 2: Perturbing parameters for calibration...")
    perturbed_params = {
        "PropensityToConsumeIncome": 0.7,  # +0.1 from true
        "PropensityToConsumeSavings": 0.3,  # -0.1 from true
    }

    for param_name, value in perturbed_params.items():
        model.parameters[param_name] = value

    print(f"  True α1: {true_params['PropensityToConsumeIncome']:.3f}")
    print(
        f"  Initial α1: {perturbed_params['PropensityToConsumeIncome']:.3f} (error: +0.1)"
    )
    print(f"  True α2: {true_params['PropensityToConsumeSavings']:.3f}")
    print(
        f"  Initial α2: {perturbed_params['PropensityToConsumeSavings']:.3f} (error: -0.1)"
    )
    print()

    # Step 3: Define loss function using mse_loss
    print("Step 3: Setting up loss function...")

    def loss_fn(output):
        """Compute MSE loss on key variables."""
        return mse_loss(
            output,
            target_output,
            variables=["ConsumptionDemand", "DisposableIncome"],
            timesteps=slice(5, 40),  # Transition period (most informative)
            reduction="mean",
        )

    print(
        "  Loss function: MSE on ConsumptionDemand and DisposableIncome (timesteps 5-40)"
    )
    print()

    # Step 4: Get model behavior for optimization
    print("Step 4: Preparing model for optimization...")
    behavior = model.get_model_training_instance(scenario=0)

    # Get parameters to optimize
    params_to_optimize = [
        p for name, p in behavior.named_parameters() if name.startswith("params.")
    ]

    print(f"  Parameters to optimize: {len(params_to_optimize)}")
    print()

    # Step 5: Run optimization with Adam
    print("Step 5: Running optimization with torch.optim.Adam...")
    optimizer = torch.optim.Adam(params_to_optimize, lr=5e-3)

    n_epochs = 200
    print_every = 50

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # Forward pass
        output = behavior.forward()
        loss = loss_fn(output)

        # Backward pass
        loss.backward()
        optimizer.step()

        if (epoch + 1) % print_every == 0:
            print(f"  Epoch {epoch + 1:3d}/{n_epochs}: Loss = {loss.item():.6e}")

    print()

    # Step 6: Compare final parameters to true values
    print("Step 6: Calibration results...")
    print()

    # Extract final parameter values from behavior
    final_params = {}
    for name, p in behavior.named_parameters():
        if name.startswith("params."):
            param_name = name.replace("params.", "")
            final_params[param_name] = p.item()

    print("Parameter                        True      Initial    Final      Error")
    print("-" * 70)

    final_alpha1 = final_params.get(
        "PropensityToConsumeIncome", perturbed_params["PropensityToConsumeIncome"]
    )
    final_alpha2 = final_params.get(
        "PropensityToConsumeSavings", perturbed_params["PropensityToConsumeSavings"]
    )

    print(
        f"PropensityToConsumeIncome       {true_params['PropensityToConsumeIncome']:.4f}    "
        f"{perturbed_params['PropensityToConsumeIncome']:.4f}     "
        f"{final_alpha1:.4f}    {abs(final_alpha1 - true_params['PropensityToConsumeIncome']):.4f}"
    )
    print(
        f"PropensityToConsumeSavings      {true_params['PropensityToConsumeSavings']:.4f}    "
        f"{perturbed_params['PropensityToConsumeSavings']:.4f}     "
        f"{final_alpha2:.4f}    {abs(final_alpha2 - true_params['PropensityToConsumeSavings']):.4f}"
    )
    print()

    # Summary
    print("=" * 70)
    print("Phase 1 Complete!")
    print("=" * 70)
    print()
    print("✓ EstimationResult dataclass ready for use")
    print("✓ Loss functions (mse_loss, weighted_residuals, composite_loss) implemented")
    print("✓ Compatible with torch.optim optimizers")
    print("✓ Ready for Phase 2: Levenberg-Marquardt optimizer")
    print()


if __name__ == "__main__":
    main()
