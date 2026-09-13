"""Action-selection controllers.

The controller is the component that turns *measurements* into a *choice* among
the six cache actions. It is the second half of the project's research question:
policies decide what is valuable, controllers decide what to do about it under
explicit constraints.

Status
------
:class:`~uniqkache.controllers.adaptive.AdaptiveController` is a **research
prototype**. Its decision rules are a hypothesis. See its module docstring.
"""

from __future__ import annotations

from uniqkache.controllers.adaptive import AdaptiveController
from uniqkache.controllers.types import (
    ActionKind,
    Constraints,
    ControllerState,
    Decision,
    Objective,
    QualityRisk,
    quality_risk_of,
)

__all__ = [
    "ActionKind",
    "AdaptiveController",
    "Constraints",
    "ControllerState",
    "Decision",
    "Objective",
    "QualityRisk",
    "quality_risk_of",
]
