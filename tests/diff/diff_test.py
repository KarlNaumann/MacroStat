import torch

from macrostat.diff import (
    JacobianAutograd,
    JacobianNumerical,
    check_model_differentiability,
)
from macrostat.models import get_model


class TestDiffLinear2D:
    def setup_method(self):
        self.model_cls = get_model("LINEAR2D")
        self.model = self.model_cls()

    def make_loss_fn(self):
        target = torch.tensor([1.0, 0.0])

        def loss_fn(output: dict[str, torch.Tensor]) -> torch.Tensor:
            y = output["State"][-1]
            return torch.nn.functional.mse_loss(y, target)

        return loss_fn

    def test_autograd_vs_numerical_close(self):
        loss_fn = self.make_loss_fn()
        auto = JacobianAutograd(self.model)
        num = JacobianNumerical(self.model, epsilon=1e-5)

        grads_auto = auto.compute(loss_fn=loss_fn, mode="rev")
        grads_num = num.compute(loss_fn=loss_fn, mode="central")

        for name in grads_auto:
            ga = grads_auto[name]
            gn = grads_num[name]
            assert ga.shape == gn.shape
            # Relative error should be modest for this simple linear system
            denom = torch.maximum(ga.abs(), gn.abs()).clamp_min(1.0)
            rel = (ga - gn).abs() / denom
            assert rel.max().item() < 1e-2

    def test_checker_reports_small_relative_errors(self):
        loss_fn = self.make_loss_fn()
        report = check_model_differentiability(
            model=self.model,
            loss_fn=loss_fn,
            scenario=0,
            rtol=1e-5,
            atol=1e-8,
            compare_forward_reverse=True,
            compare_numerical=True,
            numerical_mode="central",
            epsilon=1e-5,
        )

        assert not report.nan_or_inf
        # Forward vs reverse should agree to around 1e-6 or better
        assert report.rel_err_fwd_rev is not None
        assert report.rel_err_fwd_rev < 1e-5
        # Autograd vs numerical should agree to around 1e-2 on this toy model
        assert report.rel_err_autodiff_num is not None
        assert report.rel_err_autodiff_num < 1e-2
