"""Small experiment metrics used by the robustness study."""

from __future__ import annotations


def percent_reduction(baseline_pollution: float, portfolio_pollution: float) -> float:
    """Return pollution reduction relative to the matching no-GI baseline."""

    baseline = float(baseline_pollution)
    current = float(portfolio_pollution)
    if baseline <= 0.0:
        return 0.0 if current <= 0.0 else -1.0
    return 1.0 - current / baseline


def robustness_ratio(
    unopposed_reduction: float, attacked_reduction: float
) -> float | None:
    """Return attacked benefit divided by unopposed benefit."""

    unopposed = float(unopposed_reduction)
    if unopposed <= 0.0:
        return None
    return float(attacked_reduction) / unopposed
