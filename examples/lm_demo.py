"""
Demo script for Levenberg-Marquardt optimizer.

This script demonstrates parameter calibration using the LM optimizer:
1. Generate synthetic data from known parameters
2. Perturb parameters away from true values
3. Run LM optimization to recover the true parameters
"""

import torch

from macrostat.estimation import LevenbergMarquardt, mse_loss
from macrostat.models import get_model


def main():
    print("=" * 70)
    print("Levenberg-Marquardt Optimization Demo")
    print("=" * 70)

    # Create model
    model = get_model("GL06SIM")()

    # Step 1: Generate synthetic target data from known parameters
    print("\n[Step 1] Generating synthetic target data...")
    true_params = {
        "PropensityToConsumeIncome": 0.6,
        "PropensityToConsumeSavings": 0.4,
    }

    for name, value in true_params.items():
        model.parameters[name] = value
    print(f"  True parameters: {true_params}")

    with torch.no_grad():
        target_output = model.simulate(scenario=0)
    print("  Target data generated ✓")

    # Step 2: Perturb parameters (simulate bad initial guess)
    print("\n[Step 2] Perturbing parameters...")
    initial_params = {
        "PropensityToConsumeIncome": 0.7,
        "PropensityToConsumeSavings": 0.3,
    }

    for name, value in initial_params.items():
        model.parameters[name] = value
    print(f"  Initial guess: {initial_params}")

    # Step 3: Define loss function
    def loss_fn(output):
        return mse_loss(
            output,
            target_output,
            variables=["ConsumptionDemand", "DisposableIncome"],
            timesteps=slice(5, 40),  # Use transition dynamics
            reduction="none",  # Return residual vector for LM
        )

    # Step 4: Run LM optimization
    print("\n[Step 3] Running Levenberg-Marquardt optimization...")
    print("  (This will take ~10-20 seconds)\n")

    lm = LevenbergMarquardt(
        model,
        loss_fn,
        max_nfev=100,
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
        damping_init=1e-7,  # Smaller damping for this problem
        verbose=1,  # Print iteration progress
    )

    result = lm.optimize()

    # Step 5: Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print("\n📊 Parameter Recovery:")
    print(f"{'Parameter':<35} {'True':<12} {'Initial':<12} {'Final':<12} {'Error':<10}")
    print("-" * 70)

    for name in true_params.keys():
        true_val = true_params[name]
        initial_val = initial_params[name]
        final_val = result.params[name].item()
        error_pct = abs(final_val - true_val) / true_val * 100

        print(
            f"{name:<35} {true_val:<12.6f} {initial_val:<12.6f} "
            f"{final_val:<12.6f} {error_pct:<10.2f}%"
        )

    print("\n📈 Optimization Statistics:")
    print(f"  Final cost:              {result.cost:.6e}")
    print(f"  Optimality (gradient):   {result.optimality:.6e}")
    print(f"  Iterations:              {result.nit}")
    print(f"  Function evaluations:    {result.nfev}")
    print(f"  Jacobian evaluations:    {result.njev}")
    print(f"  Converged:               {result.success}")
    print(f"  Termination:             {result.message}")

    # Success criteria
    print("\n✅ Success Criteria:")
    max_error = max(
        abs(result.params[name].item() - true_params[name]) / true_params[name] * 100
        for name in true_params.keys()
    )
    print(f"  Maximum error:           {max_error:.2f}%")

    if max_error < 1.0:
        print("  ✓ Excellent recovery (< 1% error)")
    elif max_error < 5.0:
        print("  ✓ Good recovery (< 5% error)")
    else:
        print("  ✗ Poor recovery (> 5% error)")

    if result.cost < 1e-6:
        print("  ✓ Cost essentially zero")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
