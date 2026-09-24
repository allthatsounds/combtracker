"""Cross-species external validation of the missing-fundamental exemption.

Runs the ICASSP paper's missing-fundamental sweep (high-pass away H1..Hk,
rescore) on four taxa from the Dryad cross-species F0 benchmark
(doi:10.5061/dryad.prr4xgxw8), with three arms: the comb tracker as shipped,
the comb tracker with the exemption disabled, and pYIN.

The point is not whether the comb tracker beats pYIN on these species -- all
four record their fundamental strongly, so at k=0 it should not. The point is
whether the *exemption* buys the same thing on other taxa, at F0s spanning a
factor of 30, that it buys on elephant rumbles.

Ground truth here is a direct per-frame (time, F0) CSV rather than the
elephant corpus's hand-traced harmonic stack, so `derive_f0_curve_v2` is
substituted with an identity so that every metric downstream of it in
`pitch_tracker.evaluate.score_tracked_f0` is reused verbatim.
"""
from __future__ import annotations

import argparse, glob, json, os, sys, time, warnings
import numpy as np, pandas as pd, soundfile as sf
from scipy.signal import butter, sosfiltfilt

# Resolve the tracker package: $PITCH_TRACKER_ROOT, else the parent directory
# (the shipped reproduce/ layout puts pitch_tracker/ beside xspecies/), else
# this directory (the working layout, where it sits alongside).
_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.environ.get("PITCH_TRACKER_ROOT"),
              os.path.dirname(_HERE), _HERE):
    if _cand and os.path.isdir(os.path.join(_cand, "pitch_tracker")):
        sys.path.insert(0, _cand)
        break
else:
    sys.exit("cannot locate the pitch_tracker package; set PITCH_TRACKER_ROOT")

# `pitch_tracker.fine_contour` imports cool-frames at module level for a
# function this experiment never calls. Stub it so the package imports; nothing
# below touches the stub.
import types as _types


class _StubFinder:
    """Satisfy any `cool_frames.*` import with an empty module."""
    def find_module(self, name, path=None):
        return self if name == "cool_frames" or name.startswith("cool_frames.") else None

    def load_module(self, name):
        if name in sys.modules:
            return sys.modules[name]
        m = _types.ModuleType(name)
        m.__path__ = []          # make every stub a package
        m.__loader__ = self
        m.__getattr__ = lambda attr: None
        sys.modules[name] = m
        return m


sys.meta_path.append(_StubFinder())

import pitch_tracker.comb_f0 as CF
import pitch_tracker.evaluate as EV
import pitch_tracker.shrp as SHRP

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Species configuration.
#
# win_cycles: the elephant setting is a 1.25 s window on a ~15 Hz fundamental,
# i.e. ~19 cycles of F0. Holding cycles rather than seconds constant preserves
# the harmonic resolution the comb score depends on: sigma_hz = 0.94 / T_win
# then scales with F0 automatically, exactly as it does when the elephant
# harness shortens its window on a short call.
# --------------------------------------------------------------------------
WIN_CYCLES = 19.0
HOPS_PER_WIN = 25.0          # elephant: 50 ms hop on a 1.25 s window

SPECIES = {
    # name:            (f0_nominal_hz, f0_p5, f0_p95, k_max)
    "lions":            (150.0,  131.0,  191.0, 6),
    "spotted_hyenas":   (312.0,  228.0,  381.0, 6),
    "monk_parakeets":  (1760.0, 1374.0, 2091.0, 6),
    "long-billed_hermits": (4477.0, 3425.0, 6468.0, 3),
}


def species_params(name):
    f0n, p5, p95, kmax = SPECIES[name]
    win = WIN_CYCLES / f0n
    return dict(
        f0_nominal=f0n,
        f0_min=p5 / 3.0,          # room to make an octave error downward
        f0_max=p95 * 3.0,         # and upward -- otherwise the test is vacuous
        win_seconds=win,
        min_win_seconds=win * 0.4,
        hop_seconds=win / HOPS_PER_WIN,
        # elephant setting is a 60 Hz median on a ~15 Hz fundamental
        floor_width_hz=4.0 * f0n,
        # elephants search 8-400 Hz on a ~15 Hz F0, i.e. ~27 harmonics; the
        # scorer clips the harmonic index at 40, so a wider band in harmonic
        # terms folds high peaks onto k=40 and invents matches.
        band_hi_harmonics=27.0,
        k_max=kmax,
        k_list=(list(range(kmax + 1)) if kmax <= 3
                else [k for k in (0, 2, 4, 6) if k <= kmax]),
    )


MAX_FRAMES = 150.0
"""Cost bound. hop = max(win/25, duration/MAX_FRAMES). The elephant setting
gives ~40 frames on a 2 s call; because hop scales as 1/F0, holding win/25
would give a high-F0 species an order of magnitude more frames per second for
no gain -- the octave decision is a modal statistic over frames."""


# --------------------------------------------------------------------------
# The exemption switch.
#
# Shipped `comb_score_frame` counts empty comb lines only between k_lo (the
# lowest *observed* harmonic) and k_hi. Setting the lower bound to 1 instead
# charges the candidate for every empty line below its lowest observed peak,
# which is precisely the term the paper's ablation removes.
# --------------------------------------------------------------------------
def make_comb_score_frame(exempt: bool):
    def comb_score_frame(peak_freqs, peak_weights, f0_cands, *,
                         sigma_hz=1.5, gap_penalty=0.55, max_harmonic=40):
        n_c = len(f0_cands)
        scores = np.full(n_c, -np.inf, dtype=float)
        lowest_k = np.zeros(n_c, dtype=int)
        if peak_freqs.size == 0:
            return scores, lowest_k
        w = np.asarray(peak_weights, dtype=float)
        p = np.asarray(peak_freqs, dtype=float)
        order = np.argsort(p)
        p, w = p[order], w[order]
        total_w = float(np.sum(w))
        if total_w <= 0.0:
            return scores, lowest_k
        mean_w = float(np.mean(w))
        f = np.asarray(f0_cands, dtype=float)[:, None]
        k = np.clip(np.round(p[None, :] / f), 1, max_harmonic)
        dev = np.abs(p[None, :] - k * f)
        match = np.exp(-((dev / sigma_hz) ** 2))
        explained = np.sum(w[None, :] * match, axis=1)
        hit = match > 0.5
        valid = hit.any(axis=1) & (explained > 0.0)
        k_lo = np.where(hit, k, np.inf).min(axis=1)
        k_hi = np.where(hit, k, -np.inf).max(axis=1)
        k_hit = np.where(hit, k, 0.0)
        prev_max = np.maximum.accumulate(k_hit, axis=1)
        prev_max = np.concatenate([np.zeros((n_c, 1)), prev_max[:, :-1]], axis=1)
        dup = hit & (k == prev_max)
        n_distinct = hit.sum(axis=1) - dup.sum(axis=1)
        start = k_lo if exempt else np.ones(n_c)      # <-- the one-term change
        n_interior = np.maximum(k_hi - start + 1, 1)
        n_empty = n_interior - n_distinct
        s = (explained - gap_penalty * mean_w * n_empty) / total_w
        scores[valid] = s[valid]
        lowest_k[valid] = k_lo[valid].astype(int)
        return scores, lowest_k
    return comb_score_frame


_SHIPPED = CF.comb_score_frame


def verify_exempt_patch():
    """The exempt=True patch must reproduce the shipped function exactly."""
    rng = np.random.default_rng(0)
    pf = np.sort(rng.uniform(20, 400, 12)); pw = rng.uniform(1, 20, 12)
    cands = np.geomspace(8, 60, 300)
    a = _SHIPPED(pf, pw, cands, sigma_hz=1.5, gap_penalty=1.0)
    b = make_comb_score_frame(True)(pf, pw, cands, sigma_hz=1.5, gap_penalty=1.0)
    same = np.allclose(np.nan_to_num(a[0], neginf=-1e9),
                       np.nan_to_num(b[0], neginf=-1e9)) and np.array_equal(a[1], b[1])
    return bool(same)


# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------
def load_species(name, root="data"):
    d = os.path.join(root, name)
    wavs = sorted(glob.glob(os.path.join(d, "**", "*.wav"), recursive=True) +
                  glob.glob(os.path.join(d, "**", "*.WAV"), recursive=True))
    out = []
    for w in wavs:
        c = os.path.splitext(w)[0] + ".csv"
        if not os.path.exists(c):
            continue
        try:
            df = pd.read_csv(c)
        except Exception:
            continue
        t = pd.to_numeric(df.iloc[:, 0], errors="coerce").values
        f = pd.to_numeric(df.iloc[:, 1], errors="coerce").values
        m = np.isfinite(t) & np.isfinite(f) & (f > 0)
        if m.sum() < 5:
            continue
        out.append((os.path.relpath(w, root), w, t[m], f[m]))
    return out


def make_derived(t, f0):
    """Stand in for derive_f0_curve_v2 on a corpus that annotates F0 directly."""
    return {
        "grid_t": np.asarray(t, float),
        "f0_derived": np.asarray(f0, float),
        "c1_interp": np.asarray(f0, float),
        "n_contours": 1,
        "harmonic_number_of_contour1": 1,
        "gt_spread_cents": np.full(len(t), np.nan),
    }


def remove_low_harmonics(x, fs, f0, k):
    if k <= 0 or not f0 or f0 <= 0:
        return np.asarray(x, float)
    cut = (k + 0.5) * f0
    nyq = fs / 2.0
    if cut >= 0.95 * nyq:
        return None                      # not achievable on this clip
    sos = butter(8, cut / nyq, btype="highpass", output="sos")
    return sosfiltfilt(sos, np.asarray(x, float))


# --------------------------------------------------------------------------
# Trackers
# --------------------------------------------------------------------------
_PICK = CF.pick_peaks


def _scaled_pick_peaks(floor_width_hz):
    """`pick_peaks` hardcodes a 60 Hz noise-floor median -- four times the
    elephant fundamental. At hermit scale 60 Hz sits well inside one harmonic
    spacing, so the median tracks the harmonics themselves and no peak clears
    the 6 dB threshold. Scale it with F0 as the elephant setting implies."""
    def f(mag, freqs, *, f_lo, f_hi, threshold_db=6.0,
          floor_width_hz=floor_width_hz, max_peaks=40):
        return _PICK(mag, freqs, f_lo=f_lo, f_hi=f_hi,
                     threshold_db=threshold_db,
                     floor_width_hz=floor_width_hz, max_peaks=max_peaks)
    return f


def run_comb(x, fs, P, exempt):
    CF.comb_score_frame = make_comb_score_frame(exempt)
    CF.pick_peaks = _scaled_pick_peaks(P["floor_width_hz"])
    try:
        r = CF.estimate_f0(
            x, fs,
            f0_min=P["f0_min"], f0_max=P["f0_max"],
            win_seconds=P["win_seconds"], min_win_seconds=P["min_win_seconds"],
            hop_seconds=P["hop_seconds"],
            band_lo=P["f0_min"] * 0.8,
            band_hi=min(0.95 * fs / 2, P["band_hi_harmonics"] * P["f0_nominal"]),
        )
    finally:
        CF.comb_score_frame = _SHIPPED
        CF.pick_peaks = _PICK
    return np.asarray(r.times), np.asarray(r.f0)


def run_pyin(x, fs, P):
    import librosa
    fl = int(2 ** np.ceil(np.log2(max(256, 4 * fs / P["f0_min"]))))
    fl = min(fl, int(2 ** np.floor(np.log2(max(256, len(x))))))
    hop = max(16, int(round(P["hop_seconds"] * fs)))
    f0, _, _ = librosa.pyin(x, fmin=P["f0_min"], fmax=P["f0_max"], sr=int(fs),
                            frame_length=fl, hop_length=hop, fill_na=np.nan)
    t = librosa.times_like(f0, sr=int(fs), hop_length=hop)
    return t, f0


ARMS = {
    "comb":          lambda x, fs, P: run_comb(x, fs, P, True),
    "comb_noexempt": lambda x, fs, P: run_comb(x, fs, P, False),
    "pyin":          lambda x, fs, P: run_pyin(x, fs, P),
}


# --------------------------------------------------------------------------
def do_clip(job):
    sp, key, wav, gt_t, gt_f = job
    P = species_params(sp)
    out = []
    try:
        x, fs = sf.read(wav, dtype="float64", always_2d=False)
    except Exception:
        return sp, key, out, "unreadable"
    if x.ndim > 1:
        x = x.mean(1)
    dur = len(x) / fs
    Pc = dict(P)
    Pc["hop_seconds"] = max(P["hop_seconds"], dur / MAX_FRAMES)
    f0_ann = float(np.median(gt_f))
    derived = make_derived(gt_t, gt_f)
    max_gap = max(2.5 * Pc["hop_seconds"], 0.02)
    for k in P["k_list"]:
        xk = remove_low_harmonics(x, fs, f0_ann, k)
        if xk is None:
            continue
        for arm, fn in ARMS.items():
            try:
                tt, ff = fn(xk, fs, Pc)
                m = EV.score_tracked_f0(tt, ff, derived, max_gap_s=max_gap)
            except Exception:
                m = None
            if not m or m.get("n_scored", 0) == 0:
                continue
            out.append(dict(species=sp, clip=key, k=k, arm=arm, f0_annotated=f0_ann,
                            strict_cents=m["medae_cents_strict"],
                            oct_tol_cents=m["medae_cents_octave_tolerant"],
                            octave_correct=float(m["octave_multiple_mode"] == 1.0),
                            corr=m["corr_pearson"], coverage=m["coverage"],
                            n_scored=m["n_scored"]))
    return sp, key, out, None


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", nargs="*", default=list(SPECIES))
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--out", default="results/xspecies.json")
    args = ap.parse_args()

    assert verify_exempt_patch(), "exempt=True patch does not reproduce shipped scorer"
    EV.derive_f0_curve_v2 = lambda d: d          # the seam: d is already derived

    jobs, requested = [], {}
    for sp in args.species:
        clips = load_species(sp)
        n_all = len(clips)
        if args.limit and n_all > args.limit:      # even sample, not the first N
            idx = np.linspace(0, n_all - 1, args.limit).round().astype(int)
            clips = [clips[i] for i in sorted(set(idx.tolist()))]
        requested[sp] = dict(n_in_corpus=n_all, n_requested=len(clips))
        P = species_params(sp)
        print(f"[{sp}] {len(clips)}/{n_all} clips  f0=[{P['f0_min']:.0f},{P['f0_max']:.0f}] Hz "
              f"win={P['win_seconds']*1e3:.1f} ms floor={P['floor_width_hz']:.0f} Hz "
              f"k={P['k_list']}", flush=True)
        jobs += [(sp, key, wav, t, f) for key, wav, t, f in clips]

    rows, done, failed, t_start = [], 0, [], time.time()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with mp.Pool(args.procs) as pool:
        for sp, key, out, err in pool.imap_unordered(do_clip, jobs, chunksize=1):
            done += 1
            if err or not out:
                failed.append(f"{sp}/{key}:{err or 'no-rows'}")
            rows += out
            if done % 20 == 0:
                el = time.time() - t_start
                print(f"   {done}/{len(jobs)}  {el:.0f}s  eta {el/done*(len(jobs)-done):.0f}s",
                      flush=True)
                pd.DataFrame(rows).to_csv(args.out.replace(".json", "_rows.csv"), index=False)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out.replace(".json", "_rows.csv"), index=False)

    summary = {}
    for (sp, arm, k), g in df.groupby(["species", "arm", "k"]):
        summary.setdefault(sp, {}).setdefault(arm, {})[int(k)] = dict(
            n=int(len(g)),
            strict_cents_median=float(g["strict_cents"].median()),
            oct_tol_cents_median=float(g["oct_tol_cents"].median()),
            octave_correct=float(g["octave_correct"].mean()),
            corr_median=float(g["corr"].median()),
            coverage_median=float(g["coverage"].median()),
        )
    coverage = {}
    for sp in args.species:
        g = df[df["species"] == sp]
        coverage[sp] = dict(requested[sp],
                            n_scored=int(g["clip"].nunique()) if len(g) else 0,
                            n_rows=int(len(g)))
        coverage[sp]["all_items_scored"] = (
            coverage[sp]["n_scored"] == coverage[sp]["n_requested"])
    meta = dict(win_cycles=WIN_CYCLES, hops_per_win=HOPS_PER_WIN,
                max_frames=MAX_FRAMES,
                species={s: species_params(s) for s in args.species},
                exempt_patch_verified=True, coverage=coverage,
                n_failed=len(failed), failed=failed[:40],
                n_rows=int(len(df)), elapsed_s=round(time.time() - t_start, 1),
                dataset="doi:10.5061/dryad.prr4xgxw8")
    with open(args.out, "w") as f:
        json.dump(dict(meta=meta, summary=summary), f, indent=1)
    print(json.dumps(summary, indent=1)[:2000])
    print(f"\nwrote {args.out}  ({len(df)} rows, {time.time()-t_start:.0f}s)")


if __name__ == "__main__":
    main()
