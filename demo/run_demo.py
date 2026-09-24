#!/usr/bin/env python3
"""Five annotated rumbles, three objectives, no corpus needed.

Runs on the clips in ``demo/clips/``: real African elephant calls from the
corpus the paper is scored on, included here with permission (see DATA.md).
On each of them the lowest hand-traced harmonic is well above the fundamental,
which is the case the paper is about.

Three trackers run on each clip:

    comb                the published objective, exemption included
    comb, no exemption  the same objective charging the gap penalty from the
                        first harmonic up -- a plain two-way mismatch
    SHRP                the strongest published baseline over all calls

and each is scored against the traced contours with the shipped scorer.

    python3 demo/run_demo.py
    python3 demo/run_demo.py --plot     also writes demo/demo.png

Expected: the comb objective lands on the annotated fundamental on all five,
SHRP loses the octave (downward) on all five, and the objective without the
exemption loses it on one (EFAF2011A030.WAV_386_353, an octave up). Over the
corpus the exemption is worth 3.0 points; these five were chosen, so the
proportion here says nothing about that.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import wave

import numpy as np

# Deriving F0 from contour spacing takes medians over frames that can be empty
# at a call's edges; numpy warns on those and the reference handles them.
np.seterr(invalid="ignore")
import warnings                                                      # noqa: E402
warnings.filterwarnings("ignore", r"All-NaN slice encountered")
warnings.filterwarnings("ignore", r"Degrees of freedom <= 0")
warnings.filterwarnings("ignore", r"Mean of empty slice")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pitch_tracker import comb_f0 as CF                              # noqa: E402
from pitch_tracker import shrp as SHRP                               # noqa: E402
from pitch_tracker.evaluate import score_tracked_f0                  # noqa: E402
from pitch_tracker.groundtruth import derive_f0_curve_v2             # noqa: E402

FS = 2000.0
#: Table 1's SHRP operating point.
SHRP_A = dict(win_seconds=1.25, hop_seconds=0.05, f0_min=8, f0_max=60,
              n_f0_candidates=400, n_harmonics=6)


def read_wav(path):
    """The demo clips are 16-bit mono at 2 kHz, so stdlib ``wave`` is enough."""
    with wave.open(path, "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise SystemExit("%s: expected 16-bit mono" % path)
        fs = float(w.getframerate())
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return x.astype(float) / 32768.0, fs


def load_reference():
    with open(os.path.join(HERE, "reference.json")) as fh:
        raw = json.load(fh)
    out = {}
    for clip, contours in raw.items():
        out[clip] = [{"contourID": c["contourID"], "label": c["label"],
                      "t": np.asarray(c["t"], float),
                      "f": np.asarray(c["f"], float)} for c in contours]
    return out


def run(tracker, x, fs):
    if tracker == "shrp":
        r = SHRP.extract_shrp(x, fs, **SHRP_A)
    else:
        r = CF.estimate_f0(x, fs)
    return np.asarray(r.times, float), np.asarray(r.f0, float)


def no_exempt_scorer():
    """The same objective with the missing-fundamental exemption removed.

    ``comb_f0.comb_score_frame`` is a module-level function, so the ablation
    harness's variant can be swapped in without touching the tracker -- this is
    the ``-- exemption`` row of the paper's ablation, not a re-implementation.
    """
    from pitch_tracker.ablation import make_scorer
    return make_scorer(exempt=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plot", action="store_true", help="also write demo/demo.png")
    a = ap.parse_args()

    ref = load_reference()
    clips = sorted(ref)
    rows = []
    original = CF.comb_score_frame

    for clip in clips:
        path = os.path.join(HERE, "clips", clip)
        if not os.path.exists(path):
            raise SystemExit("missing %s" % path)
        x, fs = read_wav(path)
        contours = ref[clip]
        d = derive_f0_curve_v2(contours)
        if not d:
            raise SystemExit("%s: no derivable reference" % clip)
        f0_ref = float(np.nanmedian(np.asarray(d["f0_derived"], float)))
        lowest = min(contours, key=lambda c: float(np.nanmedian(c["f"])))
        lowest_traced = float(np.nanmedian(lowest["f"]))
        # Which harmonic the lowest traced contour is, taken from the ratio
        # itself. ``harmonic_number_of_contour1`` numbers the annotation's own
        # first contour, which is not always the lowest one in frequency.
        k1 = lowest_traced / f0_ref if f0_ref > 0 else float("nan")

        rec = {"clip": clip, "f0_ref": f0_ref, "k1": k1,
               "lowest_traced": lowest_traced, "dur": len(x) / fs}
        for name in ("comb", "comb_no_exempt", "shrp"):
            try:
                if name == "comb_no_exempt":
                    CF.comb_score_frame = no_exempt_scorer()
                    t, f = run("comb", x, fs)
                    CF.comb_score_frame = original
                else:
                    t, f = run(name, x, fs)
                s = score_tracked_f0(t, f, contours)
            finally:
                CF.comb_score_frame = original
            if s is None or not s.get("n_scored"):
                rec[name] = None
                continue
            rec[name] = {
                "f0": float(np.nanmedian(f[np.isfinite(f) & (f > 0)]))
                      if np.isfinite(f).any() else float("nan"),
                "cents": float(s["medae_cents_strict"]),
                "octave_ok": bool(abs(s["octave_multiple_mode"] - 1.0) < 1e-9),
                "multiple": float(s["octave_multiple_mode"]),
            }
        rows.append(rec)

    w = max(len(r["clip"]) for r in rows)
    print()
    print("Five annotated rumbles. 'lowest traced' is the lowest hand-traced")
    print("harmonic; on every clip here it sits well above the fundamental.")
    print("These five were CHOSEN to show the failure mode -- on each of them a")
    print("majority of the published baselines lose the octave. They are not a")
    print("random sample, so the counts below are an illustration, not a result.")
    print()
    hdr = ("%-*s  %7s %7s %5s | %s" %
           (w, "clip", "F0 ref", "lowest", "as", "   comb          no exempt      SHRP"))
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        cells = []
        for name in ("comb", "comb_no_exempt", "shrp"):
            v = r[name]
            if v is None:
                cells.append("%-14s" % "no answer")
            else:
                cells.append("%6.1f Hz %s%-3s" % (
                    v["f0"], "ok " if v["octave_ok"] else "x  ",
                    "" if v["octave_ok"] else ("x%g" % v["multiple"])))
        print("%-*s  %6.1fH %6.1fH  H%-3.0f| %s" %
              (w, r["clip"], r["f0_ref"], r["lowest_traced"], round(r["k1"]),
               "  ".join(cells)))
    print()
    for name, label in (("comb", "comb (published)"),
                        ("comb_no_exempt", "comb, no exemption"),
                        ("shrp", "SHRP")):
        ok = sum(1 for r in rows if r[name] and r[name]["octave_ok"])
        cents = [r[name]["cents"] for r in rows if r[name] and r[name]["octave_ok"]]
        print("  %-20s octave correct on %d of %d   median error %s"
              % (label, ok, len(rows),
                 ("%.1f cents" % float(np.median(cents))) if cents else "n/a"))
    print()
    print("For the measured result, the paper reports 97.0 % octave-correct "
          "for the")
    print("comb objective over 675 held-out clips (Table 1), 94.1 % with the "
          "exemption")
    print("removed (section 4.1), and 75.4 % for SHRP over the same clips.")

    if a.plot:
        make_plot(rows, ref)


def make_plot(rows, ref):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed; skipping the figure)")
        return
    fig, axes = plt.subplots(len(rows), 1, figsize=(6.2, 1.5 * len(rows)),
                             sharex=False)
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, rows):
        top = 0.0
        for c in ref[r["clip"]]:
            ax.plot(c["t"], c["f"], lw=0.8, color="0.65",
                    label="traced harmonics" if c is ref[r["clip"]][0] else None)
            top = max(top, float(np.nanmax(c["f"])))
        ax.axhline(r["f0_ref"], color="C0", lw=1.6, label="annotated $F_0$")
        for name, col, lab in (("comb", "C2", "comb"),
                               ("comb_no_exempt", "C3", "comb, no exemption")):
            if r[name]:
                ax.axhline(r[name]["f0"], color=col, ls="--", lw=1.3, label=lab)
        # show every traced harmonic: these clips are the point precisely
        # because the lowest one sits far above the fundamental.
        ax.set_ylim(0, 1.08 * top)
        ax.set_ylabel("Hz", fontsize=8)
        ax.set_title("%s   lowest traced harmonic: H%.0f at %.0f Hz"
                     % (r["clip"], round(r["k1"]), r["lowest_traced"]), fontsize=7)
        ax.tick_params(labelsize=7)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), frameon=False)
    axes[-1].set_xlabel("time (s)", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    p = os.path.join(HERE, "demo.png")
    fig.savefig(p, dpi=150)
    print("  wrote %s" % p)


if __name__ == "__main__":
    main()
