"""Human vulnerability: indoor casualties via building damage state, outdoor via direct effects.

For each population cell:
    P_fatal = f_in * sum_ds P(ds) * fatal_rate[ds]
            + (1 - f_in) * (1 - (1 - P_blast_fatal)(1 - P_burn_fatal))
and similarly for injuries (excluding fatalities).

All rates and curve parameters here are ILLUSTRATIVE defaults (HAZUS-style indoor casualty
structure; direct-blast and burn curves loosely based on Glasstone & Dolan ch. 11-12).
Replace with referenced values before quantitative use.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from effects.fragility.curves import LognormalCurve
from effects.units import cal_cm2_to_j_m2


@dataclass(frozen=True)
class CasualtyModel:
    # Indexed by damage state including 'none' first: none, light, moderate, severe, collapse
    indoor_fatal_rate: tuple[float, ...] = (0.0, 0.0, 0.001, 0.02, 0.25)
    indoor_injury_rate: tuple[float, ...] = (0.0, 0.01, 0.05, 0.30, 0.50)
    # Outdoor direct blast (overpressure, Pa) - lung damage etc. plus translation
    blast_fatal: LognormalCurve = field(default_factory=lambda: LognormalCurve(240e3, 0.3))
    blast_injury: LognormalCurve = field(default_factory=lambda: LognormalCurve(35e3, 0.4))
    # Outdoor thermal (fluence, J/m^2): fatal ~ 3rd-degree over large area; injury ~ 2nd-degree
    burn_fatal: LognormalCurve = field(
        default_factory=lambda: LognormalCurve(cal_cm2_to_j_m2(10.0), 0.3)
    )
    burn_injury: LognormalCurve = field(
        default_factory=lambda: LognormalCurve(cal_cm2_to_j_m2(5.0), 0.3)
    )

    def rates(self, state_probs, p, q, indoor_fraction):
        """Return (P_fatal, P_injured) per cell.

        state_probs: (..., n_states+1) damage-state probabilities for the cell's building stock.
        p: overpressure (Pa), q: thermal fluence (J/m^2), indoor_fraction: scalar or array.
        """
        fin = np.asarray(indoor_fraction)
        sp = np.asarray(state_probs)
        in_fatal = sp @ np.asarray(self.indoor_fatal_rate)
        in_any = sp @ (np.asarray(self.indoor_fatal_rate) + np.asarray(self.indoor_injury_rate))

        out_fatal = 1.0 - (1.0 - self.blast_fatal.prob(p)) * (1.0 - self.burn_fatal.prob(q))
        out_any = 1.0 - (1.0 - self.blast_injury.prob(p)) * (1.0 - self.burn_injury.prob(q))
        out_any = np.maximum(out_any, out_fatal)

        fatal = fin * in_fatal + (1.0 - fin) * out_fatal
        injured = fin * (in_any - in_fatal) + (1.0 - fin) * (out_any - out_fatal)
        return fatal, np.maximum(injured, 0.0)
