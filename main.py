"""gta6-trophy-rarity-forecast-01 entry: forecast GTA VI's PS5 trophy earn rates on Exophase.

Model (pure Python 3.13 standard library, deterministic, no network):

  Each case carries every read the house made of Exophase's GTA VI PS5 trophy
  page before the checkpoint: per read, the players tracked and every trophy's
  earn rate (percent of tracked players who have earned it). Exophase refreshes
  a game's counts on its own schedule, hours apart, so the page is a step
  function: long flat stretches, then a jump. The forecast is per trophy, on
  the logit scale, against x = ln(1 + hours since launch, 2026-11-19T00:00Z).

  Two regimes, chosen by whether the history has ever seen a refresh:

  * Flat history (every read identical: no refresh yet, as in the practice
    cases). The next refresh dilutes the rates: Exophase discovers a wave of
    newly tracked players who have earned far fewer trophies than the early
    cohort. Prior, not fitted to any answer: the tracked base grows about
    fivefold (GROWTH = 5) and the new cohort earns each non-universal trophy
    at about a tenth of the early cohort's rate (RHO = 0.10), blended so a
    trophy everyone already has (rate near 100%) stays near 100%. The median
    is the mixed rate (old cohort + new cohort) / grown base, with a wide
    logit spread (FLAT_SPREAD = 0.9) because the refresh's size is a guess.

  * Refresh history (the final case: ~2.5 days of reads, refreshes throughout).
    For each trophy, a recency-weighted least-squares line of logit(rate)
    against x over the distinct reads (consecutive identical reads collapsed,
    so a long flat stretch cannot dominate the fit), weights halving every 24
    hours. Slopes shrink toward the cross-trophy median slope (pseudo-count
    3), so a trophy with little movement borrows the typical growth instead
    of reporting a flat or noisy line. The median is anchored on the trophy's
    last observed rate at the last distinct read's clock (the first read that
    showed those values, i.e. the last refresh -- the page is frozen between
    refreshes, so anchoring at the checkpoint would misdate the level) and
    extended by the shrunk slope to the target's x.
    The spread combines the fit's residual scale with a random-walk term that
    grows with the horizon (0.4 logit units per sqrt day) and a floor, so a
    4.5-day extrapolation is honestly wider than a 90-minute one.

  Quantiles are the median's logit plus the normal z for each level times the
  spread, mapped back to percent and rounded to 4 decimals half up (the
  starter's rounding, matching JavaScript's Math.round).

  Backtest (see build/analysis/backtest.py; the kit ships no local grader and
  no practice answers, so the practice answers are reconstructed from the two
  aggregates the competition itself publishes for the 04:30Z refresh --
  3,868 players tracked, 4,276 trophies earned -- under two shapes,
  proportional and logit-shift, which agree closely): this model scores
  0.232 mean pinball loss on the practice against persistence's 1.401 on the
  same reconstruction (the competition states persistence at about 1.51).
  On synthetic final-like cases (2.5 days of refreshes, known truth 4.5 days
  out, three seeds) it scores 0.19 against persistence's 0.49 and the house
  curve baseline's 0.23 (see build/analysis/synth_test.py).

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0),
standard library only. Outside the grader, MAKEFUN_INPUT_DIR and
MAKEFUN_OUTPUT_DIR stand in for /input and /output. For every <id>.in in the
input directory it writes <id>.out: {"trophies": [{"id": n, "q": [q05, q25,
q50, q75, q95]}]}, one row per trophy, each q five finite numbers from 0 to
100, non-decreasing, at most 65,536 bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first (starter's pattern).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Launch instant: the game's release day begins 2026-11-19T00:00:00Z.
LAUNCH = datetime(2026, 11, 19, tzinfo=timezone.utc)

# --- flat-history (first-refresh) prior: chosen a priori, not fitted ---
# Tracked-base growth at the first refresh, and the new cohort's earn rate
# relative to the early cohort's, for a non-universal trophy.
GROWTH = 5.0
RHO = 0.10
# Logit spread in the flat regime: the refresh's size is a guess.
FLAT_SPREAD = 0.9

# --- trend regime ---
# Recency weight half-life for the log-time fit, in hours.
HALF_LIFE_HOURS = 24.0
# Pseudo-count for shrinking a trophy's slope toward the median slope.
SHRINK_PSEUDO = 3.0
# Sane bounds for a log-time slope (logit units per unit of ln(1 + hours)).
SLOPE_MIN, SLOPE_MAX = -1.5, 3.0
# Bounds for a fit's residual scale (logit units).
RESID_FLOOR, RESID_CAP = 0.15, 1.0
# Random-walk spread, logit units per square-root day of horizon.
RW_SIGMA = 0.4
# Variance floor added to every spread, so no forecast is ever overconfident.
SPREAD_VAR_FLOOR = 0.04


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    return math.log(clamp(pct) / (100 - clamp(pct)))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100 / (1 + math.exp(-lg))
    e = math.exp(lg)
    return 100 * e / (1 + e)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's Math.round(x * 10000) / 10000 rounds it.

    A whole number comes back as an int, so json.dumps writes 100 where JavaScript writes 100 (not 100.0).
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_launch(ts: datetime) -> float:
    return max(0.0, (ts - LAUNCH).total_seconds() / 3600.0)


def x_of(ts: datetime) -> float:
    """Log-time coordinate of a read: ln(1 + hours since launch)."""
    return math.log1p(hours_since_launch(ts))


def distinct_reads(history: list) -> list:
    """Collapse consecutive identical reads (same players, same rates).

    Exophase's page only changes at its own refreshes, so most reads repeat
    the previous one exactly; keeping them would let a long flat stretch
    dominate any fit. A read is kept when the players tracked or any rate
    differs from the previous kept read.
    """
    kept = []
    for read in history:
        if kept:
            prev = kept[-1]
            if (read.get("players_tracked") == prev.get("players_tracked")
                    and read.get("rates") == prev.get("rates")):
                continue
        kept.append(read)
    return kept


def quantiles_from(median_logit: float, spread: float) -> list:
    row = []
    for z in Z:
        q = round4(clamp(inv_logit(median_logit + z * spread)))
        row.append(q)
    # Rounding can only keep the order (inv_logit is increasing in z); guard anyway.
    for i in range(1, len(row)):
        if row[i] < row[i - 1]:
            row[i] = row[i - 1]
    return row


def flat_forecast(rate: float) -> list:
    """First-refresh prior for a trophy whose history never moved.

    The new cohort earns a non-universal trophy at RHO times the early
    cohort's rate; a trophy near 100% is universal, so its blend factor
    rises to 1 as the rate approaches 100% (rho_i = RHO + (1 - RHO) * r^2
    with r the rate as a fraction). The mixed rate after the base grows
    GROWTH-fold is (r + (GROWTH - 1) * q) / GROWTH.
    """
    r = clamp(rate) / 100.0
    rho_i = RHO + (1.0 - RHO) * r * r
    q = r * rho_i
    mixed = (r + (GROWTH - 1.0) * q) / GROWTH
    return quantiles_from(logit(mixed * 100.0), FLAT_SPREAD)


def weighted_line(points: list) -> tuple:
    """Recency-weighted least squares y = a + b*x over (x, y, weight).

    Returns (a, b, resid): the intercept, slope and weighted residual scale.
    With fewer than 2 distinct x values, or no variance in x, the slope is 0
    and the residual is the weighted spread of y around its mean.
    """
    if not points:
        return 0.0, 0.0, RESID_FLOOR
    sw = sum(w for _, _, w in points)
    mx = sum(w * x for x, _, w in points) / sw
    my = sum(w * y for _, y, w in points) / sw
    sxx = sum(w * (x - mx) ** 2 for x, _, w in points)
    if sxx <= 1e-12:
        resid = math.sqrt(sum(w * (y - my) ** 2 for _, y, w in points) / sw)
        return my, 0.0, min(RESID_CAP, max(RESID_FLOOR, resid))
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in points)
    b = sxy / sxx
    a = my - b * mx
    resid = math.sqrt(sum(w * (y - (a + b * x)) ** 2 for x, y, w in points) / sw)
    return a, b, min(RESID_CAP, max(RESID_FLOOR, resid))


def forecast_case(case: dict, name: str) -> dict:
    if not case.get("history"):
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    history = case["history"]
    kept = distinct_reads(history)
    horizon_days = case["horizon_minutes"] / 1440.0
    x_target = x_of(parse_ts(case["target"]))
    last_ts = parse_ts(history[-1]["at"])
    x_last = x_of(last_ts)

    # Last observed rate per trophy (last non-null read). The as-of time for
    # every trophy is the last DISTINCT read: the page is frozen between
    # refreshes, so the last read's values are the last refresh's values, and
    # anchoring them at the checkpoint's x would pretend they were measured
    # hours later than they were (the last distinct read is the first read
    # that showed them, within one 10-minute slot of the refresh itself).
    last_rate = {}
    for read in history:
        for i, rate in enumerate(read["rates"]):
            if rate is not None:
                last_rate[i] = rate
    anchor_x = x_of(parse_ts(kept[-1]["at"]))
    # A trophy never observed in any read: fall back to the median last rate.
    observed = sorted(last_rate.values())
    median_last = observed[len(observed) // 2] if observed else 1.0
    for i in range(len(trophies)):
        if i not in last_rate:
            last_rate[i] = median_last

    if len(kept) <= 1:
        # No refresh has ever been seen: the first-refresh prior, per trophy.
        rows = [{"id": t["id"], "q": flat_forecast(last_rate[i])}
                for i, t in enumerate(trophies)]
        return {"trophies": rows}

    # Trend regime: per-trophy log-time fit over the distinct reads.
    fits = {}
    slopes = []
    for i in range(len(trophies)):
        points = []
        for read in kept:
            rate = read["rates"][i]
            if rate is None:
                continue
            t = parse_ts(read["at"])
            w = 0.5 ** ((last_ts - t).total_seconds() / 3600.0 / HALF_LIFE_HOURS)
            points.append((x_of(t), logit(rate), w))
        # Distinct y values: how much slope evidence this trophy carries.
        n_distinct = len({round(y, 9) for _, y, _ in points})
        a, b, resid = weighted_line(points)
        fits[i] = (a, b, resid, n_distinct)
        if len(points) >= 3 and n_distinct >= 2:
            slopes.append(b)
    slopes.sort()
    median_slope = slopes[len(slopes) // 2] if slopes else 0.0

    rows = []
    for i, t in enumerate(trophies):
        _a, b, resid, n_distinct = fits[i]
        # Shrink the slope toward the cross-trophy median.
        b_shrunk = (n_distinct * b + SHRINK_PSEUDO * median_slope) / (n_distinct + SHRINK_PSEUDO)
        b_shrunk = min(SLOPE_MAX, max(SLOPE_MIN, b_shrunk))
        median_logit = logit(last_rate[i]) + b_shrunk * (x_target - anchor_x)
        spread = math.sqrt(resid ** 2 + (RW_SIGMA ** 2) * horizon_days + SPREAD_VAR_FLOOR)
        rows.append({"id": t["id"], "q": quantiles_from(median_logit, spread)})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
