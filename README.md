# gta6-twitch-ccu-forecast-01 — Launch-day curve forecaster

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/elpablo61/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/elpablo61/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A forecaster for the
[make.fun gta6-twitch-ccu-forecast-01 bounty](https://make.fun/bounty/gta6-twitch-ccu-forecast-01):
from the Twitch viewer count of Grand Theft Auto VI at the end of its
release day's first hour, forecast the count in each of hours 2–24 as five
quantiles (5%, 25%, 50%, 75%, 95%), scored by pinball loss on the log scale.

## Method — comparable launch curves, anchored on hour 1

A launch-day curve is anchored on its hour-1 print. Across the past
launches bundled in the case, the later hours' log-levels depend strongly
on the log hour-1 level (correlation about −0.94 at hour 20): a big hour 1
means the world already unlocked and the day stays near that level; a tiny
hour 1 means the launch wave is still coming and the day explodes. Per
hour, the forecast blends two views of the same pool of past launches:

1. **Local-linear:** weighted least squares of log viewers(hour) on
   log viewers(hour 1), Gaussian kernel on the log hour-1 distance
   (bandwidth 1.25), with Gaussian residual quantiles whose spread is
   smoothed across neighbouring hours.
2. **Ratio quantiles:** weighted empirical quantiles of
   log(viewers(hour) / viewers(hour 1)), same scale kernel
   (bandwidth 1.0), anchored on the case's hour-1 print.
3. **Blend:** 60% (1) + 40% (2) in log space.

Comparables that also launched on PC get a lower weight (0.5 in hours 2–5,
0.15 in hours 6–24): their PC unlock, hours after the console unlock,
adds a second wave a console-only launch never has. GTA VI is console
only, so the console launches in the pool carry the forecast. Any hour
the pool cannot cover falls back to the house persistence baseline.

## Validation — leave-one-out over the kit's comparables

The practice answers are sealed, so nothing was tuned on them. The model
was tuned by leave-one-out over the 37 past launches in the kit (each one
forecast from its own hour 1, scored against its own hours 2–24):

| evaluation set | house baseline | this entry |
|---|---|---|
| all 37 comparables | 0.930 | **0.382** |
| big launches (hour 1 ≥ 20k) | 0.321 | **0.249** |
| console midnight-local family | 0.368 | **0.275** |
| console-only launches | ~0.32 | **0.212** |

(Mean pinball loss on the log scale; lower is better. The house baseline
scores about 0.387 on the official practice cases.)

Rejected on the same leave-one-out: unweighted OLS (0.440), plain
ratio-quantiles (0.405), diurnal-profile weighting (0.621 — much worse),
a pre-release-buzz second regressor (no gain), and dropping PC launches
entirely (0.722 — the pool gets too small).

## Files

- `main.py` — the whole forecaster (Python 3.13, standard library only).

No network, no wall clock, no files outside the case, no randomness: the
same case always produces the same forecast. Runs in ~0.2 s a case.

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Referral

Built as [@pablo-crypto](https://make.fun/r/vky0ybyt) on make.fun.
