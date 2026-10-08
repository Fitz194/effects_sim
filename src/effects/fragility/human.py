"""Human vulnerability: indoor casualties via building damage state, outdoor via direct effects.

For each population cell:
    P_fatal = f_in * sum_ds P(ds) * fatal_rate[ds]
            + (1 - f_in) * (1 - (1 - P_blast_fatal)(1 - P_burn_fatal))
and similarly for injuries (excluding fatalities).

Sources (Glasstone & Dolan 1977, "The Effects of Nuclear Weapons", printed page numbers):
  * Indoor rates: Table 12.21 (p. 547), casualties of ~1,600 people in reinforced-concrete
    buildings in Japan versus structural damage. The book itself warns (Sec. 12.21-12.23) that the
    numbers "cannot be used to estimate casualties from the degree of structural damage". They are
    the only primary tabulation, so they are used verbatim and the caveats are in docs/DESIGN.md.
  * Outdoor direct blast: Table 12.38 (p. 551), tentative criteria for effective peak pressure.
  * Outdoor burns: Fig. 12.65 (p. 565), skin-burn probabilities for an average unshielded
    population as a function of yield and radiant exposure (yield-dependent).

Each parameter's basis is tagged in the comments: SOURCED (read directly), DERIVED (computed from
sourced values, steps given) or ASSUMED (no source; judgement).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from effects.fragility.curves import LognormalCurve
from effects.units import cal_cm2_to_j_m2, psi_to_pa


@dataclass(frozen=True)
class YieldDependentCurve:
    """Lognormal curve whose median radiant exposure grows with log10(yield).

    median(Y) = intercept + slope * log10(Y_kt)  [cal/cm^2], read as straight lines on the
    semi-log Fig. 12.65, which are accurate to the figure's reading error over 1 kt - 10 Mt.
    """

    intercept_cal_cm2: float
    slope_cal_cm2_per_decade: float
    beta: float

    def median_j_m2(self, yield_kt: float) -> float:
        cal = self.intercept_cal_cm2 + self.slope_cal_cm2_per_decade * float(np.log10(yield_kt))
        return cal_cm2_to_j_m2(cal)

    def prob(self, q_j_m2, yield_kt: float):
        return LognormalCurve(self.median_j_m2(yield_kt), self.beta).prob(q_j_m2)


@dataclass(frozen=True)
class CasualtyModel:
    # Indexed by damage state including 'none' first: none, light, moderate, severe, collapse.
    # SOURCED (Table 12.21, p. 547): killed outright 8 / 14 / 88 % and serious (hospitalised)
    # injury 14 / 18 / 11 % for light / moderate / severe structural damage; 'none' is 0.
    # ASSUMED mapping: this model's 'severe' and 'collapse' both take the book's "severe damage".
    # CAVEAT: the data come from 0.3-0.75 mile from the Hiroshima/Nagasaki bursts, so deaths include
    # early radiation and burns. Light/moderate rates probably overstate casualties at large range.
    indoor_fatal_rate: tuple[float, ...] = (0.0, 0.08, 0.14, 0.88, 0.88)
    indoor_injury_rate: tuple[float, ...] = (0.0, 0.14, 0.18, 0.11, 0.11)
    # Outdoor direct blast (incident overpressure, Pa).
    # SOURCED median: Table 12.38 lethality 50 % = 62 psi (range 50-75). DERIVED beta: threshold
    # 40 psi and 100 % at 92 psi read as the 1 % and 99 % points -> beta = ln(92/40)/(2*2.326) =
    # 0.18. Note: Table 12.38 is for *effective* peak pressure, incident + a dynamic-pressure term
    # depending on body orientation; using incident pressure underestimates blast injury close in.
    blast_fatal: LognormalCurve = field(
        default_factory=lambda: LognormalCurve(psi_to_pa(62.0), 0.18)
    )
    # DERIVED median 17 psi: eardrum rupture 50 % for adults is 15-20 psi and lung damage runs from
    # 12 psi (threshold) to 25 psi (severe), geometric mean 17.3 psi. ASSUMED beta 0.3.
    blast_injury: LognormalCurve = field(
        default_factory=lambda: LognormalCurve(psi_to_pa(17.0), 0.3)
    )
    # Outdoor thermal, yield-dependent (Fig. 12.65: average unshielded population, no evasion).
    # Median = the figure's 50 % lines: 3rd degree 6.0 + 1.21 log10(Y); 2nd 4.0 + 0.71 log10(Y).
    # Injury = 2nd-degree burns (hospital-level), SOURCED lines, ASSUMED beta 0.3 (the figure's
    # 18 %/82 % bands give 0.2-0.5 depending on how the bands are read).
    # Fatal = 3rd-degree burns: ASSUMED proxy, since the book gives no burn-lethality curve; 50 %
    # third-degree burns is not 50 % mortality, so treat burn deaths as an upper bound.
    burn_fatal: YieldDependentCurve = field(
        default_factory=lambda: YieldDependentCurve(6.0, 1.21, 0.3)
    )
    burn_injury: YieldDependentCurve = field(
        default_factory=lambda: YieldDependentCurve(4.0, 0.71, 0.3)
    )

    def rates(self, state_probs, p, q, indoor_fraction, *, yield_kt):
        """Return (P_fatal, P_injured) per cell.

        state_probs: (..., n_states+1) damage-state probabilities for the cell's building stock.
        p: overpressure (Pa), q: thermal fluence (J/m^2), indoor_fraction: scalar or array,
        yield_kt: weapon yield, which sets the burn thresholds.
        """
        fin = np.asarray(indoor_fraction)
        sp = np.asarray(state_probs)
        in_fatal = sp @ np.asarray(self.indoor_fatal_rate)
        in_any = sp @ (np.asarray(self.indoor_fatal_rate) + np.asarray(self.indoor_injury_rate))

        out_fatal = 1.0 - (1.0 - self.blast_fatal.prob(p)) * (
            1.0 - self.burn_fatal.prob(q, yield_kt)
        )
        out_any = 1.0 - (1.0 - self.blast_injury.prob(p)) * (
            1.0 - self.burn_injury.prob(q, yield_kt)
        )
        out_any = np.maximum(out_any, out_fatal)

        fatal = fin * in_fatal + (1.0 - fin) * out_fatal
        injured = fin * (in_any - in_fatal) + (1.0 - fin) * (out_any - out_fatal)
        return fatal, np.maximum(injured, 0.0)
