"""Bootstrap confidence intervals (spec §7), stdlib-only (no scipy dependency).

`paired_bootstrap_compare` is *paired*: the same reports are evaluated across
arms, so each bootstrap iteration resamples one shared set of case indices
and applies it to both arms' per-case values, rather than resampling each
arm independently (which would overstate the variance of the difference).
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _is_usable(v) -> bool:
    return v is not None and not (isinstance(v, float) and math.isnan(v))


@dataclass(frozen=True)
class BootstrapResult:
    point_estimate: float
    ci_low: float
    ci_high: float
    n_boot: int
    seed: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PairedBootstrapResult:
    metric_name: str
    arm_a: str
    arm_b: str
    diff: float  # metric(arm_a) - metric(arm_b)
    ci_low: float
    ci_high: float
    p_value: float | None  # two-sided empirical p; None if no paired cases
    n_boot: int
    seed: int
    n_paired_cases: int

    def to_dict(self) -> dict:
        return asdict(self)


def bootstrap_ci(
    values: list[float], *, seed: int, n_boot: int = 2000, ci: float = 0.95
) -> BootstrapResult:
    """Percentile bootstrap CI on the mean of `values` (e.g. per-case 0/1
    'correct' indicators for one metric). NaN/None entries are dropped before
    resampling. Fewer than 2 usable values returns a degenerate result
    (ci_low == ci_high == point_estimate) rather than raising; zero usable
    values returns NaN for all three fields."""
    usable = [v for v in values if _is_usable(v)]
    if len(usable) < 2:
        point = _mean(usable) if usable else float("nan")
        return BootstrapResult(point_estimate=point, ci_low=point, ci_high=point, n_boot=n_boot, seed=seed)

    rng = random.Random(seed)
    n = len(usable)
    point = _mean(usable)

    boot_means = []
    for _ in range(n_boot):
        resample = [usable[rng.randrange(n)] for _ in range(n)]
        boot_means.append(_mean(resample))
    boot_means.sort()

    lower_p = (1 - ci) / 2
    upper_p = 1 - lower_p
    lo = boot_means[int(lower_p * (n_boot - 1))]
    hi = boot_means[int(upper_p * (n_boot - 1))]
    return BootstrapResult(point_estimate=point, ci_low=lo, ci_high=hi, n_boot=n_boot, seed=seed)


def paired_bootstrap_compare(
    values_a: dict[str, float],
    values_b: dict[str, float],
    *,
    metric_name: str,
    arm_a: str,
    arm_b: str,
    seed: int,
    n_boot: int = 2000,
    ci: float = 0.95,
) -> PairedBootstrapResult:
    """Paired bootstrap comparison of the same per-case metric under two arms.

    `values_a`/`values_b` are `{case_id: metric_value}` for the SAME metric on
    the SAME reports under two arms. Only case_ids present in both dicts are
    used. Each bootstrap iteration resamples the shared case_id list with
    replacement and applies the SAME resampled indices to both arms, so the
    diff-of-means each iteration reflects paired variance, not independent
    variance (spec §7: "same reports evaluated across arms"). The two-sided
    empirical p-value is the fraction of bootstrap diffs on the opposite side
    of zero from the observed diff, doubled and capped at 1.0.
    """
    shared = sorted(set(values_a) & set(values_b))
    n = len(shared)
    if n == 0:
        return PairedBootstrapResult(
            metric_name=metric_name, arm_a=arm_a, arm_b=arm_b,
            diff=float("nan"), ci_low=float("nan"), ci_high=float("nan"),
            p_value=None, n_boot=n_boot, seed=seed, n_paired_cases=0,
        )

    a_vals = [values_a[c] for c in shared]
    b_vals = [values_b[c] for c in shared]
    observed_diff = _mean(a_vals) - _mean(b_vals)

    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(_mean([a_vals[i] for i in idx]) - _mean([b_vals[i] for i in idx]))
    diffs.sort()

    lower_p = (1 - ci) / 2
    upper_p = 1 - lower_p
    lo = diffs[int(lower_p * (n_boot - 1))]
    hi = diffs[int(upper_p * (n_boot - 1))]

    if observed_diff > 0:
        n_opposite = sum(1 for d in diffs if d <= 0)
    elif observed_diff < 0:
        n_opposite = sum(1 for d in diffs if d >= 0)
    else:
        n_opposite = n_boot
    p_value = min(1.0, 2 * n_opposite / n_boot)

    return PairedBootstrapResult(
        metric_name=metric_name, arm_a=arm_a, arm_b=arm_b,
        diff=observed_diff, ci_low=lo, ci_high=hi, p_value=p_value,
        n_boot=n_boot, seed=seed, n_paired_cases=n,
    )
