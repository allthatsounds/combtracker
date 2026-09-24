#!/usr/bin/env python3
"""Fold the CREPE arm into the cross-species results and report both tables.

Writes ``results/xspecies_all_rows.csv`` (the three original arms plus CREPE)
and ``results/xspecies_all.json``, whose ``summary`` block has the same shape
as ``xspecies.json`` so ``make_fig.py`` can read either.

Two denominators are reported, as in the elephant robustness table:

``octave_correct``
    over the clips that arm answered -- what the original run reported.
``octave_correct_pop``
    over every clip on which the condition was achievable at that $k$, i.e.
    the union of clips any arm scored there. A clip an arm declined to answer
    counts against it. The union, rather than a flat 100, is the right
    denominator because ``remove_low_harmonics`` returns nothing for clips
    whose cutoff would reach Nyquist -- 14 hermit clips at k=3 are not
    attempted by any arm, and no tracker should be charged for those.

    python3 merge_crepe.py
"""
from __future__ import annotations

import csv
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SPECIES = ["long-billed_hermits", "monk_parakeets", "lions", "spotted_hyenas"]
ARMS = ["comb", "comb_noexempt", "pyin", "crepe"]
LBL = {"comb": "comb + exemption", "comb_noexempt": "comb, no exemption",
       "pyin": "pYIN", "crepe": "CREPE"}


def load():
    rows = []
    for name in ("xspecies_rows.csv", "xspecies_crepe.csv"):
        p = os.path.join(RES, name)
        if not os.path.exists(p):
            raise SystemExit("missing %s" % p)
        with open(p) as fh:
            for r in csv.DictReader(fh):
                r["k"] = int(float(r["k"]))
                for f in ("strict_cents", "oct_tol_cents", "octave_correct",
                          "corr", "coverage"):
                    r[f] = float(r[f]) if r[f] not in ("", "nan") else float("nan")
                rows.append(r)
    return rows


def main():
    rows = load()
    arms_present = sorted({r["arm"] for r in rows})
    if set(arms_present) != set(ARMS):
        raise SystemExit("arms present: %s" % arms_present)

    out = os.path.join(RES, "xspecies_all_rows.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {}
    print("octave-correct, over all clips attempted at that k "
          "(answered / attempted in brackets)\n")
    for sp in SPECIES:
        ks = sorted({r["k"] for r in rows if r["species"] == sp})
        attempted = {k: len({r["clip"] for r in rows
                             if r["species"] == sp and r["k"] == k}) for k in ks}
        print("%s   attempted %s" % (sp, [attempted[k] for k in ks]))
        print("   %-18s %s" % ("k", "  ".join("%13d" % k for k in ks)))
        for a in ARMS:
            cells = []
            for k in ks:
                s = [r for r in rows if r["species"] == sp and r["arm"] == a
                     and r["k"] == k]
                if not s:
                    cells.append("%13s" % "--")
                    continue
                n_ans = len({r["clip"] for r in s})
                c = sum(r["octave_correct"] > 0.5 for r in s)
                summary.setdefault(sp, {}).setdefault(a, {})[str(k)] = {
                    "n": n_ans,
                    "n_attempted": attempted[k],
                    "octave_correct": c / n_ans,
                    "octave_correct_pop": c / attempted[k],
                    "strict_cents_median": st.median([r["strict_cents"] for r in s]),
                    "oct_tol_cents_median": st.median([r["oct_tol_cents"] for r in s]),
                    "coverage_median": st.median([r["coverage"] for r in s]),
                }
                cells.append("%6.0f%% [%3d]" % (100 * c / attempted[k], n_ans))
            print("   %-18s %s" % (LBL[a], "  ".join(cells)))
        print()

    # the headline contrast: best at k=0 against best once harmonics are gone
    print("k = 0 against k = max, octave-correct over attempted clips")
    print("%-22s %-20s %7s %7s %7s" % ("species", "arm", "k=0", "k=max", "lost"))
    for sp in SPECIES:
        ks = sorted(int(k) for k in summary[sp]["comb"])
        for a in ARMS:
            v = summary[sp][a]
            a0 = 100 * v[str(ks[0])]["octave_correct_pop"]
            a1 = 100 * v[str(ks[-1])]["octave_correct_pop"]
            print("%-22s %-20s %6.0f%% %6.0f%% %+7.0f" % (sp, LBL[a], a0, a1, a1 - a0))
        print()

    with open(os.path.join(RES, "xspecies_all.json"), "w") as fh:
        json.dump({"summary": summary, "arms": ARMS,
                   "denominator_note": "octave_correct is over clips the arm "
                                       "answered; octave_correct_pop is over "
                                       "every clip attempted at that k"},
                  fh, indent=1)
    print("wrote results/xspecies_all_rows.csv and results/xspecies_all.json")


if __name__ == "__main__":
    main()
