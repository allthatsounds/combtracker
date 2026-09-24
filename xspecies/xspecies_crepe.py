#!/usr/bin/env python3
"""Add a CREPE arm to the cross-species benchmark of ``xspecies.py``.

Everything about the experiment is reused from ``xspecies.py`` by import --
the species parameters, the harmonic-removal filter, the ground-truth seam,
the scorer -- so the CREPE rows are produced under exactly the conditions the
comb and pYIN rows were. The clip set is not re-sampled either: it is read
back from ``results/xspecies_rows.csv``, so the new column is scored on the
same 100 clips per species as the existing ones.

The one thing CREPE needs that the other trackers do not is a frequency
shift, and here it runs the other way from the elephant corpus. CREPE's
decoder is defined over 32.7--1975 Hz. The four taxa need:

    lions       43.7-- 573 Hz   inside      -> no shift
    hyenas      76  --1143 Hz   inside      -> no shift
    parakeets  458  --6273 Hz   too high    -> x1/4
    hermits   1142  --19404 Hz  too high    -> x1/16

A shift is applied the same way the elephant harness applies its x8: the
waveform is resampled to ``16000 / SH`` and then *declared* to be at 16 kHz,
which multiplies every frequency CREPE sees by ``SH`` and divides every
duration by it. Nothing is decimated and no information is discarded; the
estimate is mapped back by ``f/SH`` and ``t*SH``. SH = 1 is the native
condition, and which species used which is recorded in the output.

The hop is set so that each clip yields the same number of frames the other
arms get (``MAX_FRAMES``), rather than CREPE's default, so the modal octave
statistic is computed over a comparable number of votes.

    python3 xspecies_crepe.py [--species ...] [--procs 2]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import types
import warnings

import numpy as np
import pandas as pd
import soundfile as sf

warnings.filterwarnings("ignore")

os.environ.setdefault("PITCH_TRACKER_ROOT", os.path.dirname(os.path.abspath(__file__)))
import xspecies as XS                                           # noqa: E402
import pitch_tracker.evaluate as EV                             # noqa: E402

# CREPE's decoder range, from torchcrepe.
CREPE_FMIN, CREPE_FMAX = 32.71, 1975.0
CREPE_RATE = 16000.0

# Frequency multiplier applied to what CREPE sees. Chosen per species as the
# largest power of two that brings [f0_min, f0_max] inside the decoder range;
# 1.0 means the native condition.
SHIFT = {
    "lions": 1.0,
    "spotted_hyenas": 1.0,
    "monk_parakeets": 0.25,
    "long-billed_hermits": 0.0625,
}


def check_shifts():
    """Every species' search range must land inside CREPE's decoder range."""
    bad = []
    for sp, sh in SHIFT.items():
        P = XS.species_params(sp)
        lo, hi = P["f0_min"] * sh, P["f0_max"] * sh
        if lo < CREPE_FMIN or hi > CREPE_FMAX:
            bad.append("%s: [%.1f, %.1f] Hz at SH=%g is outside [%.1f, %.1f]"
                       % (sp, lo, hi, sh, CREPE_FMIN, CREPE_FMAX))
    return bad


_TORCH = {}


def _crepe(x, fs_declared, fmin, fmax, hop_samples):
    """torchcrepe at the settings the paper's Table 1 arm uses."""
    if not _TORCH:
        if "torchaudio" not in sys.modules:
            try:
                import torchaudio                               # noqa: F401
            except Exception:
                stub = types.ModuleType("torchaudio")
                stub.__version__ = "stub (unused)"
                stub.load = stub.save = None
                sys.modules["torchaudio"] = stub
        import torch
        import torchcrepe
        torch.set_num_threads(1)
        _TORCH["torch"], _TORCH["tc"] = torch, torchcrepe
    torch, torchcrepe = _TORCH["torch"], _TORCH["tc"]
    # torchcrepe.predict is not reproducible run to run; seed each call,
    # as vendor/run_external_baselines._crepe now does.
    torch.manual_seed(0)
    audio = torch.tensor(np.ascontiguousarray(x, dtype=np.float32)).unsqueeze(0)
    f0, per = torchcrepe.predict(
        audio, int(fs_declared), hop_length=int(hop_samples),
        fmin=max(fmin, CREPE_FMIN), fmax=min(fmax, CREPE_FMAX),
        model="full", batch_size=256, device="cpu", return_periodicity=True)
    f0 = f0.squeeze(0).numpy().astype(float)
    p = per.squeeze(0).numpy().astype(float)
    f0[p < 0.21] = np.nan                    # torchcrepe's suggested threshold
    t = np.arange(len(f0)) * float(hop_samples) / float(fs_declared)
    return t, f0


def run_crepe(x, fs, P, sh):
    """CREPE on one clip, with the species' shift applied and undone."""
    # Resample so that declaring CREPE_RATE multiplies every frequency by sh.
    x16 = EV.resample_linear(np.asarray(x, float), float(fs), CREPE_RATE / sh)
    hop = max(16, int(round(P["hop_seconds"] / sh * CREPE_RATE)))
    t, f0 = _crepe(x16, CREPE_RATE, P["f0_min"] * sh, P["f0_max"] * sh, hop)
    return np.asarray(t) * sh, np.asarray(f0) / sh


def do_clip(job):
    sp, key, wav, gt_t, gt_f = job
    P = XS.species_params(sp)
    sh = SHIFT[sp]
    out = []
    try:
        x, fs = sf.read(wav, dtype="float64", always_2d=False)
    except Exception:
        return sp, key, out, "unreadable"
    if x.ndim > 1:
        x = x.mean(1)
    dur = len(x) / fs
    Pc = dict(P)
    Pc["hop_seconds"] = max(P["hop_seconds"], dur / XS.MAX_FRAMES)
    f0_ann = float(np.median(gt_f))
    derived = XS.make_derived(gt_t, gt_f)
    max_gap = max(2.5 * Pc["hop_seconds"], 0.02)
    for k in P["k_list"]:
        xk = XS.remove_low_harmonics(x, fs, f0_ann, k)
        if xk is None:
            continue
        try:
            tt, ff = run_crepe(xk, fs, Pc, sh)
            m = EV.score_tracked_f0(tt, ff, derived, max_gap_s=max_gap)
        except Exception as exc:                                # noqa: BLE001
            print("    %s/%s k=%d crepe: %r" % (sp, key, k, exc), flush=True)
            m = None
        if not m or m.get("n_scored", 0) == 0:
            continue
        out.append(dict(species=sp, clip=key, k=k, arm="crepe",
                        f0_annotated=f0_ann,
                        strict_cents=m["medae_cents_strict"],
                        oct_tol_cents=m["medae_cents_octave_tolerant"],
                        octave_correct=float(m["octave_multiple_mode"] == 1.0),
                        corr=m["corr_pearson"], coverage=m["coverage"],
                        n_scored=m["n_scored"]))
    return sp, key, out, None


def wanted_clips(path):
    """The clip keys already scored, so the new arm uses the same sample."""
    keys = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            keys.setdefault(r["species"], set()).add(r["clip"])
    return keys


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", nargs="*", default=list(XS.SPECIES))
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--rows", default="results/xspecies_rows.csv")
    ap.add_argument("--out", default="results/xspecies_crepe.csv")
    a = ap.parse_args()

    bad = check_shifts()
    if bad:
        raise SystemExit("shift factors do not bring every species into "
                         "CREPE's range:\n  " + "\n  ".join(bad))
    EV.derive_f0_curve_v2 = lambda d: d          # same seam as xspecies.main

    want = wanted_clips(a.rows)
    jobs = []
    for sp in a.species:
        clips = [c for c in XS.load_species(sp) if c[0] in want.get(sp, ())]
        P = XS.species_params(sp)
        sh = SHIFT[sp]
        print("[%s] %d clips (of %d already scored)  search %.0f-%.0f Hz  "
              "SH=%g -> CREPE sees %.0f-%.0f Hz  k=%s"
              % (sp, len(clips), len(want.get(sp, ())), P["f0_min"],
                 P["f0_max"], sh, P["f0_min"] * sh, P["f0_max"] * sh,
                 P["k_list"]), flush=True)
        if len(clips) != len(want.get(sp, ())):
            raise SystemExit("%s: matched %d of %d scored clips -- the audio "
                             "does not line up with the existing rows"
                             % (sp, len(clips), len(want.get(sp, ()))))
        jobs += [(sp, key, wav, t, f) for key, wav, t, f in clips]

    rows, done, t0 = [], 0, time.time()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with mp.Pool(a.procs) as pool:
        for sp, key, out, err in pool.imap_unordered(do_clip, jobs, chunksize=1):
            done += 1
            rows += out
            if done % 20 == 0 or done == len(jobs):
                el = time.time() - t0
                print("   %d/%d  %.0fs  eta %.0fs"
                      % (done, len(jobs), el, el / done * (len(jobs) - done)),
                      flush=True)
                pd.DataFrame(rows).to_csv(a.out, index=False)
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)

    summary = {}
    for (sp, k), g in df.groupby(["species", "k"]):
        summary.setdefault(sp, {})[int(k)] = dict(
            n=int(len(g)),
            strict_cents_median=float(g["strict_cents"].median()),
            oct_tol_cents_median=float(g["oct_tol_cents"].median()),
            octave_correct=float(g["octave_correct"].mean()),
            octave_correct_of_100=float(g["octave_correct"].sum() / 100.0),
            coverage_median=float(g["coverage"].median()))
    with open(a.out.replace(".csv", ".json"), "w") as fh:
        json.dump({"shift": SHIFT, "crepe_range_hz": [CREPE_FMIN, CREPE_FMAX],
                   "summary": summary, "n_rows": int(len(df)),
                   "elapsed_s": round(time.time() - t0, 1)}, fh, indent=1)
    print(json.dumps(summary, indent=1))
    print("\nwrote %s (%d rows, %.0f s)" % (a.out, len(df), time.time() - t0))


if __name__ == "__main__":
    main()
