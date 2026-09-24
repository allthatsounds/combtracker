# Cross-species external validation of the missing-fundamental exemption

Run 2026-09-09. `xspecies.py` + `results/`. Dataset: Best et al. 2025,
doi:10.5061/dryad.prr4xgxw8 (CC0), four taxa, 100 clips each, all scored,
0 failures.

## What this run is

The ICASSP paper's missing-fundamental sweep (high-pass away H1…H*k*,
rescore against the same annotation), run on four other taxa with three arms:
the comb tracker as shipped, the comb tracker with the exemption disabled, and
pYIN. It answers the paper's own open item — the sweep on this corpus is a
*characterisation*; on another corpus it is a *validation*.

## How the elephant configuration was scaled

Nothing was tuned per species. Three quantities were scaled from the elephant
setting by the ratio that defines them there, and everything else left alone:

| quantity | elephant | rule | why |
|---|---|---|---|
| analysis window | 1.25 s on ~15 Hz | **19 cycles of nominal F0** | `sigma_hz = 0.94/T_win` then holds the same fraction of F0, so harmonic resolution is preserved |
| noise-floor median | 60 Hz on ~15 Hz | **4 × nominal F0** | at 60 Hz fixed, the median sits inside one harmonic spacing for a 4.5 kHz species and flattens every peak — no peaks clear the 6 dB threshold |
| analysis band | 8–400 Hz on ~15 Hz | **27 harmonics** | the scorer clips the harmonic index at 40; a wider band in harmonic terms folds high peaks onto k=40 and invents matches |

Candidate range is [p5/3, p95×3] of each species' annotated F0, so an octave
error remains *possible* in both directions — otherwise the test is vacuous.
Hop is `max(win/25, duration/150)`, a cost bound only; the octave decision is
a modal statistic over frames.

The `exempt=True` code path is asserted at run time to reproduce the shipped
`comb_score_frame` bit-for-bit (`verify_exempt_patch`), so the two comb arms
differ in exactly one term: whether empty comb lines below the lowest observed
peak are charged.

## Result: octave-correct rate, 100 clips per species

| species | F0 | arm | k=0 | k=2 | k=4 | k=6 |
|---|---|---|---|---|---|---|
| lions | 150 Hz | **comb** | **96 %** | **85 %** | **73 %** | **71 %** |
| | | comb, no exemption | 63 % | 5 % | 0 % | 0 % |
| | | pYIN | 98 % | 34 % | 43 % | 11 % |
| monk parakeets | 1760 Hz | **comb** | **72 %** | **60 %** | **32 %** | **31 %** |
| | | comb, no exemption | 71 % | 9 % | 0 % | 0 % |
| | | pYIN | 33 % | 5 % | 0 % | 10 % |
| spotted hyenas | 312 Hz | comb | 57 % | 52 % | 47 % | 50 % |
| | | comb, no exemption | 62 % | 45 % | 29 % | 25 % |
| | | pYIN | 71 % | 39 % | 31 % | 22 % |
| long-billed hermits | 4477 Hz | comb | 21 % | 9 %¹ | 8 %¹ | — |
| | | comb, no exemption | 84 % | 7 %¹ | 3 %¹ | — |
| | | pYIN | 90 % | 18 %¹ | 14 %¹ | — |

¹ hermits use k = 0,1,2,3 (only ~5 harmonics below Nyquist); the columns shown
are k=2 and k=3.

## Description

**It reproduces on two of four taxa, and the two failures are diagnostic.**

- **Lions (150 Hz) and monk parakeets (1.8 kHz)** reproduce the elephant
  result. The exemption is worth 33 and 1 points with the fundamental present,
  and **71–80** and **31–51** points once harmonics are removed. pYIN starts
  best on lions (98 %) and collapses to 11 %.
- **Hyenas and hermits fail, and both fail *downward*** — onto F0/2 (median
  tracked/annotated ratio 0.52) and F0/3 (0.35). Downward failure is what the
  *gap term* prevents, not what the exemption addresses.
- **The cause is measured, not guessed.** Hermits have only **4.7 harmonics
  below Nyquist** and 8 peaks per frame; hyena frames **saturate the 40-peak
  cap** at 40 peaks per frame. In neither case does the gap term have the
  interior evidence it needs to reject a subharmonic. This is the same failure
  mode the paper already states as a limitation ("short faint calls with two or
  three peaks above the floor — too few peaks for the gap term"), now confirmed
  on independent taxa.
- Consequently, on hermits the exemption **costs 63 points** (21 % against
  84 %). The exemption is a prior that the low lines are missing; where the
  stack is too short for the gap term to adjudicate, the prior is unsupported.

**The honest headline is an operating envelope, not a clean win:** the
mechanism transfers where the band holds roughly ten or more resolved
harmonics, and does not where it holds five or where the peak set saturates.

## Caveat on the hyena arm

Saturating the 40-peak cap is a property of the *configuration*, not of the
species. It was left at the elephant value deliberately — retuning `max_peaks`
per species until the result improved is exactly the tuned comparison this
programme warns about. It should be reported as a configuration limit, and a
principled peak budget (e.g. proportional to the harmonics in band) is the
obvious follow-up.

## Redrawing the figure without downloading anything

The per-clip results ship, so the tables above and Figure 2 rebuild from them:

```
pip install numpy matplotlib
python3 merge_crepe.py
python3 make_fig.py --1col
```

## Getting the audio, to re-run the sweep

Download Best et al. 2025 from `doi:10.5061/dryad.prr4xgxw8` (CC0) and lay it
out as `data/<species>/`, one directory per species named exactly as the table
above spells it — `lions`, `spotted_hyenas`, `monk_parakeets`,
`long-billed_hermits` — with each `.wav` beside its annotation `.csv` of the
same stem. Subdirectories are searched, so the archive's own nesting is fine.
Then:

```
python3 xspecies.py                 # three arms: comb, comb without the exemption, pYIN
python3 xspecies_crepe.py           # adds the CREPE arm, needs torch + torchcrepe
python3 merge_crepe.py
python3 make_fig.py --1col
```

`data/` is git-ignored, so a download here will not be committed.

## Files

- `xspecies.py` — the whole experiment for the first three arms,
  self-contained apart from the `pitch_tracker` package it imports.
- `xspecies_crepe.py` — the CREPE arm, scored on the same clips, read back
  from `results/xspecies_rows.csv`.
- `merge_crepe.py` — folds the arms together, prints the tables, writes
  `results/xspecies_all_rows.csv` and `results/xspecies_all.json`.
- `make_fig.py` — draws Figure 2 from `results/xspecies_all.json`.
- `results/xspecies.json` — summary, per-species parameters, coverage block.
- `results/xspecies_rows.csv` — 4,260 per-(clip, k, arm) rows.
- `results/xspecies_crepe.csv` — the same rows for the CREPE arm.
- `results/xspecies_all.json` — four-arm summary, what the figure reads.
