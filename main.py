#!/usr/bin/env python3
"""Roblox Up-and-Coming top-10 forecaster (make.fun bounty
roblox-up-and-coming-forecast-01).

Reads $MAKEFUN_INPUT_DIR/history.ndjson + task.json and writes
$MAKEFUN_OUTPUT_DIR/forecast.json: per-place probabilities for the top
10 places plus a top-10 guess in order, scored by log score over the
72 hours after the deadline.

Method: Monte Carlo rank diffusion in latent-appeal space
(appeal = -log rank), 1500 seeded paths, 72 hourly steps,
Student-t(df=12) noise, phantom entrants, 10% persistence blend.

Drift regression (OLS on 4115 consecutive-pair transitions from the
46-snapshot archive, R² = 0.120):
  E[da/h] = -0.032983
           + 0.446186 * appeal_slope_96h   (trailing 96h, shrunk, capped ±0.03)
           + 0.206887 * logplaying_slope_30h (trailing 30h, shrunk, clipped ±0.05)
           + 0.010780 * log(rank)
           - 0.002263 * log(chart_age_h)
           + 0.004446 * lpr               (log(playing) − median log(playing))
  scaled by 1.2.

Standard library only, no network, deterministic (fixed seed).
Two runs produce byte-identical output.
"""

import json
import math
import os
import random
from datetime import datetime, timedelta

INPUT_DIR  = os.environ.get("MAKEFUN_INPUT_DIR",  "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# ── tunables ─────────────────────────────────────────────────────────────────

SEED               = 20261113
N_SIMS             = 1500
HORIZON_HOURS      = 72
SIGMA              = 0.10          # Student-t scale per √hour (known games)
PHANTOM_SIGMA      = 0.35          # entrants diffuse faster
T_DF               = 12
DRIFT_SCALE        = 1.2
PERSIST_W          = 0.10
DRIFT_WINDOW_H     = 96.0
GPLAY_WINDOW_H     = 30.0
DRIFT_SHRINK_K     = 2.0
DRIFT_CAP          = 0.030         # |appeal slope| per hour
GPLAY_CAP          = 0.050         # |log-playing slope| per hour
ENTRANT_DRIFT      = 0.012
ENTRANT_HALFLIFE_H = 24.0
SMOOTH_EPS         = 0.02
P_FLOOR            = 0.0015
P_CAP              = 0.65
PLACE_SUM_CAP      = 0.95

# Full-archive OLS regression coefficients (fit on 4115 transitions, R²=0.120)
REG_CONST    = -0.032983
REG_SLOPE    =  0.446186   # appeal slope 96h
REG_GPLAY    =  0.206887   # log-playing slope 30h
REG_LOG_RANK =  0.010780   # log(rank)
REG_LOG_AGE  = -0.002263   # log(chart-age hours)
REG_LPR      =  0.004446   # log(playing) − median log(playing)

# ── helpers ───────────────────────────────────────────────────────────────────

def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _slope(xs, ys, k=DRIFT_SHRINK_K):
    """Shrunk OLS slope; xs in hours."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom <= 0:
        return 0.0
    s = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    return s * n / (n + k)

# ── load history ──────────────────────────────────────────────────────────────

def load_history(input_dir):
    snaps = []
    with open(os.path.join(input_dir, "history.ndjson"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                snaps.append(json.loads(line))
    snaps.sort(key=lambda s: s["at"])
    task = {}
    try:
        with open(os.path.join(input_dir, "task.json"), encoding="utf-8") as f:
            task = json.load(f)
    except OSError:
        pass
    return snaps, task

# ── feature extraction + drift estimation ─────────────────────────────────────

def estimate_params(snaps):
    """Compute per-game drift, entry rate/ranks, last chart."""
    series      = {}   # uid -> [(t, rank)]
    play_series = {}   # uid -> [(t, playing)]
    first_seen  = {}   # uid -> t
    entry_ranks = []

    for idx, snap in enumerate(snaps):
        t = parse_time(snap["at"])
        for row in snap.get("up_and_coming") or []:
            uid = row["universe_id"]
            series.setdefault(uid, []).append((t, row["rank"]))
            if row.get("playing"):
                play_series.setdefault(uid, []).append((t, row["playing"]))
            if uid not in first_seen:
                first_seen[uid] = t
                if idx > 0:
                    entry_ranks.append(row["rank"])

    t_end    = parse_time(snaps[-1]["at"])
    t_start  = t_end - timedelta(hours=DRIFT_WINDOW_H)
    g_start  = t_end - timedelta(hours=GPLAY_WINDOW_H)

    last_chart = {
        r["universe_id"]: (r["rank"], r.get("name") or "")
        for r in snaps[-1]["up_and_coming"]
    }

    # Median log-playing of the last chart (lpr reference level)
    lps = sorted(
        math.log(max(r.get("playing") or 1, 1))
        for r in snaps[-1]["up_and_coming"]
        if r.get("playing")
    )
    med_log_play = lps[len(lps) // 2] if lps else 0.0

    drifts = {}
    for uid, obs in series.items():
        # Appeal slope (trailing 96h, shrunk n/(n+2), capped ±0.03)
        recent = [(t, rk) for t, rk in obs if t >= t_start]
        if len(recent) < 3:
            recent = obs[-3:]
        slope = 0.0
        if len(recent) >= 2:
            xs = [(t - recent[0][0]).total_seconds() / 3600.0 for t, _ in recent]
            ys = [-math.log(rk) for _, rk in recent]
            slope = min(max(_slope(xs, ys), -DRIFT_CAP), DRIFT_CAP)

        # Log-playing slope (trailing 30h, clipped ±0.05)
        trail = [(t, p) for t, p in play_series.get(uid, []) if t >= g_start]
        gplay = 0.0
        if len(trail) >= 3:
            xs = [(t - trail[0][0]).total_seconds() / 3600.0 for t, _ in trail]
            ys = [math.log(max(p, 1)) for _, p in trail]
            gplay = min(max(_slope(xs, ys), -GPLAY_CAP), GPLAY_CAP)

        # Playing level vs chart median (lpr)
        last_play = play_series.get(uid, [[None, 0]])[-1][1]
        lpr = (math.log(max(last_play, 1)) - med_log_play) if last_play else 0.0

        age_h = max((t_end - first_seen[uid]).total_seconds() / 3600.0, 1.0)
        rank  = last_chart.get(uid, (obs[-1][1], ""))[0]

        drift = (
            REG_CONST
            + REG_SLOPE    * slope
            + REG_GPLAY    * gplay
            + REG_LOG_RANK * math.log(rank)
            + REG_LOG_AGE  * math.log(age_h)
            + REG_LPR      * lpr
        ) * DRIFT_SCALE
        drifts[uid] = drift

    # Entry rate (entrants per hour observed in the history)
    span_h = 0.0
    if len(snaps) >= 2:
        span_h = (parse_time(snaps[-1]["at"]) - parse_time(snaps[0]["at"])
                  ).total_seconds() / 3600.0
    entry_rate = min(max(len(entry_ranks) / span_h, 0.05), 1.5) if span_h > 0 else 0.5

    return {
        "drifts":      drifts,
        "entry_rate":  entry_rate,
        "entry_ranks": sorted(entry_ranks) or [80, 87, 90, 95, 97],
        "last_chart":  last_chart,
    }

# ── Monte Carlo simulation ────────────────────────────────────────────────────

def simulate(params, places, horizon_hours, n_sims=N_SIMS, seed=SEED):
    """
    Returns (counts, total_slots, n_games).
    counts[uid][place] = number of (sim, hour) pairs where uid was at place.
    total_slots = n_sims * horizon_hours.
    """
    rng         = random.Random(seed)
    last_chart  = params["last_chart"]
    drifts      = params["drifts"]
    entry_rate  = params["entry_rate"]
    entry_ranks = params["entry_ranks"]

    # Games sorted by current rank
    games = [
        [uid, -math.log(rank), drifts.get(uid, 0.0)]
        for uid, (rank, _name) in sorted(last_chart.items(), key=lambda kv: kv[1][0])
    ]
    n_games = len(games)

    # Number of phantom entrants spread over the whole window
    n_phantoms = min(max(int(round(entry_rate * horizon_hours)), 25), 70)

    def t_noise():
        """Exact Student-t(T_DF) via N(0,1) / sqrt(chi2/df)."""
        z    = rng.gauss(0.0, 1.0)
        chi2 = 2.0 * rng.gammavariate(T_DF / 2.0, 1.0)
        return z / math.sqrt(chi2 / T_DF)

    counts      = {uid: [0] * (places + 1) for uid, _, _ in games}
    total_slots = n_sims * horizon_hours

    for _ in range(n_sims):
        # Copy initial state
        state = [[g[0], g[1], g[2]] for g in games]

        # Phantom entrants: entry hour uniform over window, appeal from
        # empirical entry-rank distribution (± small appeal noise)
        phantoms = []
        for _ in range(n_phantoms):
            r = entry_ranks[rng.randrange(len(entry_ranks))]
            phantoms.append([
                rng.uniform(0.0, horizon_hours),   # entry_hour
                -math.log(r) + rng.gauss(0.0, 0.3),  # initial appeal
                0.0,                               # age_in_chart (hours)
            ])

        for hour in range(1, horizon_hours + 1):
            scored = []

            # Step known games
            for g in state:
                g[1] += g[2] + t_noise() * SIGMA
                scored.append((g[1], g[0]))

            # Step phantoms that have entered; score active ones
            for ph in phantoms:
                if ph[0] <= hour:
                    ph[2] += 1.0
                    decay  = 0.5 ** (ph[2] / ENTRANT_HALFLIFE_H)
                    ph[1] += ENTRANT_DRIFT * decay + t_noise() * PHANTOM_SIGMA
                    scored.append((ph[1], -1))   # uid=-1 → unnamed

            scored.sort(key=lambda x: -x[0])

            for place in range(1, min(places, len(scored)) + 1):
                uid = scored[place - 1][1]
                if uid >= 0:
                    counts[uid][place] += 1

    return counts, total_slots, n_games

# ── build forecast probabilities ──────────────────────────────────────────────

def build_forecast(counts, total_slots, n_games, places, last_chart):
    probs = []
    for place in range(1, places + 1):
        rows = []
        for uid, arr in counts.items():
            p = arr[place] / total_slots if total_slots else 0.0
            # Smooth with a flat prior so no plausible game is at zero
            p = (1.0 - SMOOTH_EPS) * p + SMOOTH_EPS * (1.0 / max(n_games, 1))
            if p >= P_FLOOR:
                rows.append((uid, min(p, P_CAP)))
        total = sum(p for _, p in rows)
        if total > PLACE_SUM_CAP:
            scale = PLACE_SUM_CAP / total
            rows  = [(uid, p * scale) for uid, p in rows]
        rows.sort(key=lambda x: (-x[1], x[0]))
        for uid, p in rows:
            probs.append({"universe_id": uid, "place": place, "p": round(p, 6)})
    return {"probabilities": probs}

# ── persistence forecast & blend ─────────────────────────────────────────────

def persistence_forecast(last_chart, places):
    """Starter baseline: 0.5 on current-place game, 0.3 on rank-2 for
    place 1, 0.15 on immediate neighbours elsewhere."""
    order = [
        uid for uid, (_r, _n) in
        sorted(last_chart.items(), key=lambda kv: kv[1][0])
    ]
    rows = {}
    for place in range(1, places + 1):
        if place - 1 >= len(order):
            break
        rows[(order[place - 1], place)] = 0.5
        if place == 1:
            if len(order) > 1:
                rows[(order[1], 1)] = 0.3
        else:
            if place - 2 >= 0:
                rows[(order[place - 2], place)] = 0.15
            if place < len(order):
                rows[(order[place], place)] = 0.15
    return rows


def blend_persist(forecast, last_chart, places, w=PERSIST_W):
    """Blend MC forecast (weight 1-w) with persistence forecast (weight w).
    Derives top-10 guess greedily by highest probability per place."""
    persist = persistence_forecast(last_chart, places)
    mc      = {(r["universe_id"], r["place"]): r["p"]
               for r in forecast["probabilities"]}

    out  = []
    seen = set()

    for (uid, place), p in mc.items():
        blended = round((1.0 - w) * p + w * persist.get((uid, place), 0.0), 6)
        out.append({"universe_id": uid, "place": place, "p": blended})
        seen.add((uid, place))

    # Add persistence-only entries not covered by MC
    for (uid, place), p in persist.items():
        if (uid, place) not in seen and p > 0:
            out.append({"universe_id": uid, "place": place,
                        "p": round(w * p, 6)})

    out.sort(key=lambda r: (r["place"], -r["p"]))

    # Greedy top-10: highest-probability game per place, no repeats
    by_place = {}
    for row in out:
        by_place.setdefault(row["place"], []).append(row)

    top10 = []
    used  = set()
    for place in range(1, places + 1):
        for row in by_place.get(place, []):
            if row["universe_id"] not in used:
                top10.append(row["universe_id"])
                used.add(row["universe_id"])
                break

    # Fallback: fill from last chart order if top10 is short
    if len(top10) < places:
        for uid, (_rank, _name) in sorted(last_chart.items(),
                                          key=lambda kv: kv[1][0]):
            if uid not in used:
                top10.append(uid)
                used.add(uid)
            if len(top10) == places:
                break

    return {"probabilities": out, "top10": top10[:places]}

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    snaps, task = load_history(INPUT_DIR)

    places  = int(task.get("places", 10))
    window  = task.get("window") or {}
    horizon = int(window.get("slots", HORIZON_HOURS))
    if window.get("slot_minutes"):
        horizon = int(round(horizon * window["slot_minutes"] / 60.0))
    horizon = min(max(horizon, 1), HORIZON_HOURS * 4)

    params             = estimate_params(snaps)
    counts, ts, ng     = simulate(params, places, horizon)
    forecast           = build_forecast(counts, ts, ng, places, params["last_chart"])
    forecast           = blend_persist(forecast, params["last_chart"], places)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, task.get("output", "forecast.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(forecast, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
