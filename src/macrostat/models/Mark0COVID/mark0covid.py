# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Karl Naumann-Woleske
r"""Mark0COVID model class for the heterogeneous-agent Mark-0 ABM.

References:
- Gualdi, S., Tarzia, M., Zamponi, F., Bouchaud, J.-P. (2015). "Tipping
  points in macroeconomic agent-based models". JEDC 50, 29-61.
- Bouchaud, J.-P., Gualdi, S., Tarzia, M., Zamponi, F. (2018). "Optimal
  inflation target: insights from an agent-based model". Economics: The
  Open-Access, Open-Assessment E-Journal 12 (2018-15), 1-37.
- Sharma, D., Bouchaud, J.-P., Gualdi, S., Tarzia, M., Zamponi, F. (2021).
  "V-, U-, L-, or W-shaped economic recovery after COVID-19: insights from
  an agent-based model". PLOS ONE 16(3), e0247823.
"""

import logging

from macrostat.core.model import Model
from macrostat.models.Mark0COVID.behavior import BehaviorMark0COVID
from macrostat.models.Mark0COVID.parameters import ParametersMark0COVID
from macrostat.models.Mark0COVID.scenarios import ScenariosMark0COVID
from macrostat.models.Mark0COVID.variables import VariablesMark0COVID

logger = logging.getLogger(__name__)


class Mark0COVID(Model):
    r"""Mark-0 heterogeneous-agent ABM with the COVID-extension parameter
    set.

    A closed-economy ABM with :math:`N` firms, one representative household,
    one commercial bank, and one central bank. Each macro period executes 24
    ordered phases (see :class:`BehaviorMark0COVID`). The forward pass is
    differentiable end-to-end via :meth:`Behavior.tanhmask` smoothing of all
    parameter-dependent boundary conditions and frozen noise buffers drawn
    in :meth:`BehaviorMark0COVID.initialize`.

    The default scenario reproduces the Gualdi et al. (2015) "full
    employment" regime. :class:`ScenariosMark0COVID` registers the three
    additional regimes (endogenous crises, unstable, full collapse) via
    additive shocks to :math:`R` and :math:`\Theta`.
    """

    version = "Mark0COVID"

    def __init__(
        self,
        parameters: ParametersMark0COVID | None = None,
        variables: VariablesMark0COVID | None = None,
        scenarios: ScenariosMark0COVID | None = None,
        *args,
        **kwargs,
    ):
        if parameters is None:
            parameters = ParametersMark0COVID()
        if variables is None:
            variables = VariablesMark0COVID(parameters=parameters)
        if scenarios is None:
            scenarios = ScenariosMark0COVID(parameters=parameters)

        super().__init__(
            parameters=parameters,
            variables=variables,
            scenarios=scenarios,
            behavior=BehaviorMark0COVID,
            *args,
            **kwargs,
        )
