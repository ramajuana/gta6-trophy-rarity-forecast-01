"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).

Reads one case per run from the input directory and writes one forecast file
per case to the output directory: five quantiles of each trophy's earn rate
(percent of Exophase-tracked players holding it) at the case's target.

Model, in one paragraph: an earn rate is a ratio, holders over tracked
players, and the two move for different reasons. Tracked players grow as
more profiles sync (adoption); holders grow as players actually play and
reach each trophy. So the program splits every rate into holders H and
players N, fits each side in log-time (log H and log N against
log(1 + hours since release), which is close to linear for a launch week),
extrapolates both to the target, and rebuilds the rate as 100*H/N. The
forecast is then forced to be count-consistent: the holders across trophies
must add up to the independently forecast total trophies earned, so the
per-trophy forecasts are rescaled to that total. Spreads come from the
fits' residuals, the extrapolation distance, and the size of the jumps
between the page's refreshes in the history. When the history has never
shown a refresh (every read identical, as on launch night), the program
falls back to a first-refresh prior calibrated on the one refresh the kit's
own rules document (tracked players 638 -> 3868, trophies earned 943 ->
4276, the near-universal trophy staying at 100%): the page at the target
is assumed to show that refresh, with the new holders shared out in
proportion to how many each trophy already had.

Standard library only, deterministic: no clock, no randomness, no network.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817,
     1.6448536269514722]

# --- documented first refresh, from the kit's own rules text -----------------
# Launch night: the page sat unchanged (638 players tracked, 943 trophies
# earned) from the first probe through the 04:00Z read, then refreshed by
# the 04:30Z read: 3868 players, 4276 trophies earned, every rate but the
# one at 100% lower. These two points calibrate the flat-history prior.
DOC_N0 = 638.0
DOC_T0 = 943.0
DOC_N1 = 3868.0
DOC_T1 = 4276.0
DOC_T_HOURS = 4.5  # hours after release the refreshed page reflects


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit_of(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (the way the house baseline rounds)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def release_time(case):
    """Midnight UTC on the game's release date."""
    raw = case.get("game", {}).get("release_date", "2026-11-19")
    try:
        day = datetime.fromisoformat(raw).replace(tzinfo=timezone.utc)
    except ValueError:
        day = datetime(2026, 11, 19, tzinfo=timezone.utc)
    return day


def ols(xs, ys, ws=None):
    """Least squares line y ~ a + b*x. Returns (a, b, resid_std) or None."""
    n = len(xs)
    if n < 2:
        return None
    if ws is None:
        ws = [1.0] * n
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return (my, 0.0, 0.0)
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    dof = max(n - 2, 1)
    sd = math.sqrt(sum(r * r for r in resid) / dof) if n > 2 else 0.0
    return (a, b, sd)


def median(vals):
    vals = sorted(vals)
    n = len(vals)
    if n == 0:
        return 0.0
    if n % 2:
        return vals[n // 2]
    return 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def distinct_snapshots(case):
    """History compressed to reads where anything changed (a refresh).

    Returns a list of dicts {h, N, T, rates} with h = hours since release,
    in time order. Consecutive identical reads carry no information.
    """
    t0 = release_time(case)
    snaps = []
    prev = None
    for read in case["history"]:
        at = parse_time(read["at"])
        h = (at - t0).total_seconds() / 3600.0
        key = (read["players_tracked"], read["trophies_earned"],
               tuple(read["rates"]))
        if key == prev:
            continue
        prev = key
        snaps.append({
            "h": h,
            "N": float(read["players_tracked"]),
            "T": float(read["trophies_earned"]),
            "rates": list(read["rates"]),
        })
    return snaps


def quantiles_from(median_pct, sigma):
    lg = logit_of(median_pct)
    out = []
    for z in Z:
        q = inv_logit(lg + z * sigma)
        out.append(round4(min(100.0, max(0.01, q))))
    for i in range(1, 5):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def trend_forecast(case, snaps):
    """Holders/players log-time forecast. Returns (medians, sigmas) or None
    when the history is too thin to fit anything.

    The stable quantities are fitted, not the raw rates: log players and
    log total trophies against log(1 + hours since release), and each
    trophy's SHARE of all trophies earned (log share against log-time).
    Shares move slowly and are bounded, so their extrapolation is far
    steadier than extrapolating raw holder counts; the rate is then
    rebuilt as 100 * share * total / players, which is count-consistent
    by construction (the shares are renormalised to sum to one).
    """
    if len(snaps) < 3:
        return None
    trophies = case["trophies"]
    n_tr = len(trophies)
    target = parse_time(case["target"])
    t0 = release_time(case)
    h_tgt = (target - t0).total_seconds() / 3600.0
    x_tgt = math.log1p(max(h_tgt, 0.0))

    xs = [math.log1p(max(s["h"], 0.0)) for s in snaps]
    x_last = xs[-1]
    dx = max(x_tgt - x_last, 0.0)

    # --- players: power-law adoption, recent reads weighted more --------
    # Weights ramp 1 -> 4 across the snapshots: the current growth regime
    # is what extrapolates, not the launch-night transient.
    n_snaps = len(snaps)
    ws = [1.0 + 3.0 * (j / max(n_snaps - 1, 1)) ** 2 for j in range(n_snaps)]
    fit_n = ols(xs, [math.log(max(s["N"], 1.0)) for s in snaps], ws)
    if fit_n is None:
        return None
    n_hat = math.exp(fit_n[0] + fit_n[1] * x_tgt)

    # --- mean trophies per player: linear in time on recent snapshots ---
    # m(t) = T/N is U-shaped (dilution first, then play accumulation), and
    # its recent slope is the honest signal for where it goes next. A
    # log-log line for T underestimates it because m accelerates.
    m_vals = [s["T"] / max(s["N"], 1.0) for s in snaps]
    h_vals = [s["h"] for s in snaps]
    k = min(6, n_snaps)
    fit_m = ols(h_vals[-k:], m_vals[-k:])
    h_last = h_vals[-1]
    m_last = m_vals[-1]
    if fit_m is not None:
        slope_m = fit_m[1]
        # Damp the rise a little: easy trophies saturate, so the slope
        # cannot hold perfectly for the whole horizon.
        rise = slope_m * (h_tgt - h_last)
        m_hat = m_last + 0.9 * rise if rise > 0 else m_last + rise
    else:
        m_hat = m_last
    # Floor: nearly every tracked player holds the first-mission trophy,
    # so the mean cannot stay below ~1 for long. Cap: no launch-week
    # player averages more than 8 trophies by day 7.
    m_hat = min(8.0, max(0.95, m_hat))
    t_hat = m_hat * n_hat

    # --- per-trophy shares of all trophies earned -------------------------
    # Pool slopes by tier so rare trophies borrow strength. Fits are
    # recency-weighted, like the players fit.
    w_by_x = {}
    for j, x in enumerate(xs):
        w_by_x[x] = ws[j]
    tier_slopes = {}
    slopes = []
    for i, tr in enumerate(trophies):
        pts = []
        for s, x in zip(snaps, xs):
            r = s["rates"][i]
            if r is None or s["T"] <= 0:
                continue
            share = (r * s["N"] / 100.0) / s["T"]
            if share > 0:
                pts.append((x, math.log(share)))
        if len(pts) >= 3:
            f = ols([p[0] for p in pts], [p[1] for p in pts],
                    [w_by_x[p[0]] for p in pts])
            if f is not None:
                slopes.append((i, f[1]))
                tier_slopes.setdefault(tr["tier"], []).append(f[1])
    tier_med = {k: median(v) for k, v in tier_slopes.items()}
    global_med = median([b for _, b in slopes]) if slopes else 0.0

    # Empirical-Bayes slope model: a trophy's share slope depends on how
    # big its share already is. Easy trophies saturate (their share
    # falls as players move on); hard trophies take off late, so a small
    # share today predicts a steep rise. Regress the fitted slopes on
    # the trophies' current log-shares and use the line as the prior
    # for every trophy, including ones with no usable history.
    reg = None
    if len(slopes) >= 5:
        reg_pts = []
        for i, tr in enumerate(trophies):
            pts = []
            for s, x in zip(snaps, xs):
                r = s["rates"][i]
                if r is None or s["T"] <= 0:
                    continue
                share = (r * s["N"] / 100.0) / s["T"]
                if share > 0:
                    pts.append((x, math.log(share)))
            if len(pts) >= 3:
                f = ols([p[0] for p in pts], [p[1] for p in pts],
                        [w_by_x[p[0]] for p in pts])
                if f is not None:
                    reg_pts.append((pts[-1][1], f[1]))
        if len(reg_pts) >= 5:
            rfit = ols([p[0] for p in reg_pts], [p[1] for p in reg_pts])
            if rfit is not None and rfit[1] < 0:
                reg = (rfit[0], rfit[1])

    def prior_slope(logshare):
        if reg is not None:
            return min(max(reg[0] + reg[1] * logshare, -0.5), 3.0)
        return global_med

    # Cycle noise: how much log-rates move between consecutive refreshes.
    jumps = []
    for a, b in zip(snaps, snaps[1:]):
        for i in range(n_tr):
            ra, rb = a["rates"][i], b["rates"][i]
            if ra is None or rb is None or ra <= 0 or rb <= 0:
                continue
            jumps.append(abs(logit_of(rb) - logit_of(ra)))
    cyc = median(jumps) * 1.4826 if jumps else 0.15
    cyc = min(max(cyc, 0.05), 0.6)

    medians = []
    sigmas = []
    log_shares = []
    no_history = {}
    for i, tr in enumerate(trophies):
        pts = []
        for s, x in zip(snaps, xs):
            r = s["rates"][i]
            if r is None or s["T"] <= 0:
                continue
            share = (r * s["N"] / 100.0) / s["T"]
            if share > 0:
                pts.append((x, math.log(share)))
        if len(pts) >= 3:
            f = ols([p[0] for p in pts], [p[1] for p in pts],
                    [w_by_x[p[0]] for p in pts])
            a, b, sd = f
            # Shrink the slope halfway toward the slope prior (from the
            # slope-vs-share regression, else the tier median), and damp
            # it over long extrapolations: shares saturate.
            prior = prior_slope(pts[-1][1]) if reg is not None else \
                tier_med.get(tr["tier"], global_med)
            b_use = 0.5 * b + 0.5 * prior
            b_use = b_use / (1.0 + 0.35 * dx)
            wpts = [w_by_x[p[0]] for p in pts]
            sw = sum(wpts)
            mx = sum(w * p[0] for w, p in zip(wpts, pts)) / sw
            my = sum(w * p[1] for w, p in zip(wpts, pts)) / sw
            a_use = my - b_use * mx
            ls_hat = a_use + b_use * x_tgt
            resid = sd
        elif pts:
            # Anchor on the last observed share, drift at the prior rate.
            prior = prior_slope(pts[-1][1]) if reg is not None else \
                tier_med.get(tr["tier"], global_med)
            b_use = prior / (1.0 + 0.35 * dx)
            ls_hat = pts[-1][1] + b_use * dx
            resid = 0.25
        else:
            # Never seen a holder: seed half a holder's share and let it
            # grow at the prior rate for a trophy that small (the
            # regression says small shares rise steeply). The rebuild
            # caps the resulting rate at 1%: a trophy nobody held by the
            # checkpoint rarely reaches 1% by the target.
            t_last = snaps[-1]["T"]
            seed_ls = math.log(0.5 / max(t_last, 1.0))
            prior = prior_slope(seed_ls) if reg is not None else \
                tier_med.get(tr["tier"], global_med)
            b_use = max(prior, 0.0) / (1.0 + 0.35 * dx)
            ls_hat = seed_ls + b_use * dx
            no_history[i] = True
            resid = 0.9
        log_shares.append(ls_hat)
        # Spread: fit residual + slope uncertainty over the extrapolation
        # + one refresh-cycle of noise. Wider for thin data.
        sig = math.sqrt(0.10 ** 2 + resid ** 2 + (0.22 * dx) ** 2 + cyc ** 2)
        sigmas.append(min(max(sig, 0.12), 1.5))

    # --- renormalise shares, rebuild rates --------------------------------
    top = max(log_shares)
    weights = [math.exp(ls - top) for ls in log_shares]
    wsum = sum(weights)
    for i, tr in enumerate(trophies):
        share = weights[i] / wsum
        rate = 100.0 * share * t_hat / max(n_hat, 1.0)
        if i in no_history:
            rate = min(rate, 1.0)
        # Near the top the logit is hypersensitive: a trophy at 92% does
        # not jump to 100% in one horizon, it closes part of the gap.
        # Cap a high rate at closing half the remaining gap from its
        # last observed rate.
        last_r = None
        for s in reversed(snaps):
            if s["rates"][i] is not None:
                last_r = s["rates"][i]
                break
        if last_r is not None and last_r >= 80.0:
            rate = min(rate, last_r + 0.5 * (100.0 - last_r))
        medians.append(min(max(rate, 0.01), 99.99))
    return medians, sigmas


def refresh_prior_forecast(case, snaps):
    """Flat history: no refresh ever observed. Assume the page at the
    target shows the first refresh, calibrated on the documented one."""
    trophies = case["trophies"]
    last = snaps[-1] if snaps else case["history"][-1]
    n_last = float(last["N"] if isinstance(last, dict) and "N" in last
                   else last["players_tracked"])
    rates_last = last["rates"]
    t_last = float(last["T"] if isinstance(last, dict) and "T" in last
                   else last["trophies_earned"])
    target = parse_time(case["target"])
    t0 = release_time(case)
    h_tgt = max((target - t0).total_seconds() / 3600.0, 0.0)

    # Power-law adoption through the documented refresh (t=4.5h).
    beta_n = math.log(DOC_N1 / DOC_N0) / math.log1p(DOC_T_HOURS)
    n_hat = DOC_N0 * (1.0 + h_tgt) ** beta_n
    # Mean trophies per player: 1 + excess decaying as a power law,
    # calibrated to pass through the documented refresh.
    m0 = DOC_T0 / DOC_N0
    m1 = DOC_T1 / DOC_N1
    gamma = math.log((m0 - 1.0) / (m1 - 1.0)) / math.log1p(DOC_T_HOURS)
    m_hat = 1.0 + (m0 - 1.0) * (1.0 + h_tgt) ** (-gamma)
    t_hat = m_hat * n_hat

    # Split the new holders: every new player earns the near-universal
    # trophy; the rest go to the others in proportion to (holders + 0.5).
    holders = []
    for i, tr in enumerate(trophies):
        r = rates_last[i]
        holders.append(0.0 if r is None else r * n_last / 100.0)
    uni = None
    for i, r in enumerate(rates_last):
        if r is not None and r >= 99.9:
            uni = i
            break
    d_n = max(n_hat - n_last, 0.0)
    d_t = max(t_hat - t_last, 0.0)
    new_h = list(holders)
    if uni is not None:
        new_h[uni] = holders[uni] + d_n
        d_t -= d_n
    rest = [i for i in range(len(trophies)) if i != uni]
    wsum = sum(holders[i] + 0.5 for i in rest)
    for i in rest:
        new_h[i] = holders[i] + max(d_t, 0.0) * (holders[i] + 0.5) / wsum
    medians = []
    sigmas = []
    for i, tr in enumerate(trophies):
        rate = 100.0 * new_h[i] / max(n_hat, 1.0)
        if i == uni:
            rate = 100.0
        medians.append(min(max(rate, 0.01), 100.0))
        # Wide: the refresh jump is a guess, and the split is a prior.
        # Wider still for trophies with almost no holders yet.
        sig = 0.55 + 0.30 / math.sqrt(new_h[i] + 1.0)
        sigmas.append(min(sig, 1.2))
    return medians, sigmas


def forecast_case(case):
    if case["levels"] != LEVELS:
        raise ValueError("levels are not the expected five")
    if not case["history"]:
        raise ValueError("case has no history")
    snaps = distinct_snapshots(case)
    if len(snaps) >= 3:
        got = trend_forecast(case, snaps)
        if got is not None:
            medians, sigmas = got
        else:
            medians, sigmas = refresh_prior_forecast(case, snaps)
    else:
        medians, sigmas = refresh_prior_forecast(case, snaps)
    rows = []
    for tr, med, sig in zip(case["trophies"], medians, sigmas):
        rows.append({"id": tr["id"], "q": quantiles_from(med, sig)})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out = OUTPUT_DIR / (path.stem + ".out")
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    main()
