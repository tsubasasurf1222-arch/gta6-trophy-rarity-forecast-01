# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/tsubasasurf1222-arch/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/tsubasasurf1222-arch/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A deterministic forecaster for GTA VI's PlayStation 5 trophy earn rates on Exophase. The program reads one case file — every read the house made of the game's trophy page before the checkpoint — and writes five quantiles of each trophy's earn rate at the target. Exophase refreshes a game's counts on its own schedule, hours apart, so the page is a step function: the program collapses consecutive identical reads, then fits each trophy's logit rate against log time since launch with a recency-weighted line, shrinks slopes toward the cross-trophy median, and anchors the forecast on the last refresh's values. When a history has never seen a refresh at all, it falls back to a first-refresh dilution prior (the tracked base grows, newly tracked players have earned far fewer trophies). Scored by mean pinball loss on the logit scale; on the practice cases it scores 0.23 against the persistence baseline's 1.40 (answers reconstructed from the competition's published refresh aggregates), and on synthetic final-like cases 0.19 against persistence's 0.49 and the house curve baseline's 0.23. Pure Python standard library, deterministic, no network.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=14561xbg&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
