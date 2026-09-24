# Harmonic-comb $F_0$ estimation for infrasonic field recordings with absent fundamentals

The pitch tracker from the ICASSP 2027 submission, a five-clip demo, and the
cross-species run behind Figure 2.

African elephant rumbles have fundamentals between roughly 12 and 30 Hz, often
below the usable band of a field recording. Conventional estimators lock onto
the second harmonic. This objective matches harmonic combs against interpolated
spectral peaks, charges a candidate for comb lines it predicts where nothing is
observed, and **waives that charge below the lowest observed peak** — so a
candidate whose own fundamental is genuinely unrecorded is not penalised for it,
while a subharmonic, which predicts an empty line *between* every observed pair,
still is.

## Demo

```
pip install numpy
python3 demo/run_demo.py
```

Nothing else is needed. It runs three objectives on five annotated rumbles and
scores each against the traced contours:

```
clip                                F0 ref  lowest    as |    comb          no exempt      SHRP
-----------------------------------------------------------------------------------------------
AWE21-MixPre-433.WAV_a0311_63.wav    15.3H   30.5H  H2  |   15.5 Hz ok        15.5 Hz ok        10.2 Hz x  x0.5
EFAF2011A030.WAV_396_366.wav         16.9H  101.6H  H6  |   16.8 Hz ok        16.8 Hz ok         8.8 Hz x  x0.5
```

`lowest` is the lowest hand-traced harmonic and `as` which harmonic it is — 30
to 102 Hz against fundamentals of 15 to 20 Hz, which is the case the method is
for. `--plot` also writes `demo/demo.png`.

Those five clips were **chosen** to show the failure mode, so this illustrates
the mechanism rather than measuring anything. The measured result is Table 1 of
the paper: 97.0 % octave-correct over 675 held-out clips against 75.4 % for the
strongest baseline over the same clips, at a median error of 16.2 cents.

## Using the tracker

```python
from pitch_tracker import comb_f0
r = comb_f0.estimate_f0(x, fs)          # x mono; the defaults are the paper's
r.times, r.f0                           # 50 ms grid, Hz, NaN where unvoiced
```

The defaults are the published configuration: 8–60 Hz over 1200 log-spaced
candidates, a 1.25 s window, 50 ms hop, gap penalty 1.0, transition weight 12.
For another species, scale the analysis window, the noise-floor width and the
band by the ratio that defines them here — `xspecies/` does exactly that for
four species with no per-species tuning.

## Every file in this repository

```
pitch_tracker/
  comb_f0.py         the objective and the Viterbi decoding; estimate_f0 lives here
  shrp.py            SHRP (Sun 2002), the baseline the demo contrasts against
  ablation.py        make_scorer(exempt=False) — the objective with the
                     missing-fundamental exemption removed, as the paper ablated it
  groundtruth.py     recovers F0 from the spacing of hand-traced harmonic contours
  evaluate.py        the scorer: cents error, octave multiple, coverage

demo/
  run_demo.py        runs comb, comb-without-exemption and SHRP on the five clips
  reference.json     the traced contours for those clips
  clips/             five rumbles, 2 kHz mono

xspecies/
  xspecies.py        the cross-species run: comb, comb-without-exemption, pYIN,
                     over four species from the Dryad benchmark
  xspecies_crepe.py  adds the CREPE arm to the same clips
  merge_crepe.py     folds the arms together, prints the tables, writes
                     results/xspecies_all.json
  make_fig.py        draws Figure 2 from that summary
  results/           per-clip results, so the figure redraws with no audio
  README.md          where to put the Dryad download
```

To redraw Figure 2 from what ships, with no audio at all:

```
pip install numpy matplotlib
cd xspecies
python3 merge_crepe.py
python3 make_fig.py --1col
```

## What is not here

Due to distribution restrictions on the elephant sounds, this package contains
the comb tracker, a demo and the cross-species run. It is not a reproduction
package: the corpus, the 2 kHz cache, the per-clip results behind Tables 1 and
2, and the evaluation harness that produced them are not here. `DATA.md` says
what they are.

## Notes

- `pitch_tracker` needs numpy and nothing else, and imports no frame library.
- **CREPE is not reproducible run to run.** `torchcrepe.predict` returns
  different $f_0$ on identical input and `torch.manual_seed` does not fix it —
  across 220 clips a fresh seeded run matched stored seeded rows on zero of
  them. Octave decisions are unaffected and reproduce exactly; median cents
  errors are stable only to a few tenths.
- The clip-level octave verdict is a plurality over a clip's frames, which is a
  stable estimator only for a tracker that answers many frames. The comb
  objective has no clip within one frame of flipping; CREPE, which answers a
  median of five frames per clip, has 48.

## Licence

EUPL-1.2, the same licence as
[cool-frames](https://github.com/allthatsounds/cool-frames); `LICENSE` is the
text, copied from that project so the terms are byte-identical. The EUPL is
reciprocal: a derivative carries the EUPL or one of the licences its Appendix
lists as compatible.

Copyright 2026 Clara Hollomey.

The licence covers the code. It does not cover the five demo recordings, which
remain the corpus custodians' and are included by permission for demonstration,
nor the typeset IEEE version of the paper.
