# Data

## What ships here

**`demo/clips/`** — five African elephant rumbles from the corpus the paper is
scored on, included with the permission of its custodians. Mono, 16-bit, 2 kHz,
which is the rate every tracker in the paper sees; about 92 kB and 22 seconds
in total.

**`demo/reference.json`** — the hand-traced harmonic contours for those five
clips as `(contourID, label, t, f)`. These are the annotation, not a
fundamental: the lowest traced contour sits at 30 to 102 Hz against
fundamentals of 15 to 20 Hz. `pitch_tracker.groundtruth` recovers $F_0$ from
the spacing between contours, which is what the demo scores against.

The five were chosen to show the failure mode the paper is about, not sampled
at random: on each of them a majority of the published baselines lose the
octave. They illustrate the mechanism and measure nothing.

**`xspecies/results/`** — per-clip results for the cross-species benchmark, so
the figure and the tables can be redrawn without downloading any audio:

| file | what it is | produced by |
|---|---|---|
| `xspecies_rows.csv` | one row per clip per arm, three arms | `xspecies.py` |
| `xspecies_crepe.csv` | the same for the CREPE arm | `xspecies_crepe.py` |
| `xspecies.json` | summary of the three-arm run | `xspecies.py` |
| `xspecies_all.json` | summary of all four arms, read by the figure | `merge_crepe.py` |

## What does not ship

**The elephant corpus.** 809 annotated clips, 8.8 GB at the original rates,
described in `stoeger2014call`. Not ours to redistribute; access is by
arrangement with its custodians. The five demo clips are a permitted exception.

**The 2 kHz cache and the per-clip results behind Tables 1 and 2.** The
evaluation harness that produced them — the robustness sweeps, the baseline
runners, the table builders — is not in this repository either. This is the
tracker, its demo and the cross-species run.

**The cross-species audio.** Best et al. 2025, `doi:10.5061/dryad.prr4xgxw8`,
CC0 and freely downloadable. `xspecies/README.md` says where to put it.
