# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Karl Naumann-Woleske
"""KirmansAnts model class for the Kirman ant recruitment SDE.

Reference: Kirman, A. (1993). "Ants, Rationality, and Recruitment".
The Quarterly Journal of Economics, 108(1), 137-156.
"""

import logging

from macrostat.core.model import Model
from macrostat.models.KirmansAnts.behavior import BehaviorKirmansAnts
from macrostat.models.KirmansAnts.parameters import ParametersKirmansAnts
from macrostat.models.KirmansAnts.scenarios import ScenariosKirmansAnts
from macrostat.models.KirmansAnts.variables import VariablesKirmansAnts

logger = logging.getLogger(__name__)


class KirmansAnts(Model):
    """Kirman ant recruitment model in the large-N continuous limit.

    The fraction :math:`x_t \\in [0, 1]` of ants at source A evolves under
    spontaneous switching at rate :math:`\\rho` and herding at rate
    :math:`\\mu`. The stationary distribution is
    :math:`\\text{Beta}(\\rho/\\mu, \\rho/\\mu)`. The ratio :math:`\\rho/\\mu`
    determines the regime: unimodal at :math:`x=1/2` for
    :math:`\\rho/\\mu > 1`, bimodal with mass at the boundaries for
    :math:`\\rho/\\mu < 1`.

    Phase 1 mirrors the abmstat reference implementation
    (``packages/abmstat/abmstat/models/kirmanants.py``): Euler-Maruyama
    with boundary rejection. The model is non-differentiable; constructing
    the behavior with ``differentiable=True`` raises ``RuntimeError``.
    """

    version = "KirmansAnts"

    def __init__(
        self,
        parameters: ParametersKirmansAnts | None = None,
        variables: VariablesKirmansAnts | None = None,
        scenarios: ScenariosKirmansAnts | None = None,
        *args,
        **kwargs,
    ):
        if parameters is None:
            parameters = ParametersKirmansAnts()
        if variables is None:
            variables = VariablesKirmansAnts(parameters=parameters)
        if scenarios is None:
            scenarios = ScenariosKirmansAnts(parameters=parameters)

        super().__init__(
            parameters=parameters,
            variables=variables,
            scenarios=scenarios,
            behavior=BehaviorKirmansAnts,
            *args,
            **kwargs,
        )
