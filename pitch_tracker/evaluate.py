"""Scoring a tracked F0(t) against the annotated ground truth.

Metrics follow pitch-tracking convention: error in cents, a 50-cent gross-error
threshold, and an *octave-tolerant* variant that takes the best-fitting
harmonic multiple of the reference independently per frame.  The gap between
the strict and octave-tolerant numbers is the octave/harmonic-lock diagnostic;
the per-clip modal multiple says which harmonic a tracker settled on.

When two multiples tie for the mode, the tie goes to the smallest of them.
This is the rule every published number was computed with.  It is not neutral
in direction -- a 1/2-vs-1 tie scores as a downward error, a 1-vs-2 tie as
correct -- so ``octave_mode_tied`` flags the clips where it decided anything.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from .groundtruth import derive_f0_curve, derive_f0_curve_v2

__all__ = [
    "cents_error",
    "best_octave_cents_error",
    "load_wav",
    "resample_linear",
    "modal_multiple",
    "score_tracked_f0",
    "OCTAVE_MULTIPLES",
]

#: Multiples searched by :func:`best_octave_cents_error`.
OCTAVE_MULTIPLES = (0.25, 1 / 3, 0.5, 1, 2, 3, 4)


def cents_error(pred, ref):
    """``1200 * log2(pred / ref)``, elementwise."""
    pred = np.asarray(pred, dtype=float)
    ref = np.asarray(ref, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1200.0 * np.log2(pred / ref)


def best_octave_cents_error(pred, ref, multiples=OCTAVE_MULTIPLES):
    """Smallest signed cents error over harmonic multiples of ``ref``.

    Returns ``(best_error, multiple_used)``.  This is the most generous
    reading of a tracker's output: it forgives a different octave in every
    single frame.  An estimator that still fails here is not merely
    octave-shifted.
    """
    pred = np.asarray(pred, dtype=float)
    ref = np.asarray(ref, dtype=float)
    errs = np.stack([cents_error(pred, ref * m) for m in multiples], axis=0)
    idx = np.nanargmin(np.abs(errs), axis=0)
    best = errs[idx, np.arange(len(pred))]
    return best, np.asarray(multiples)[idx]


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def load_wav(path) -> tuple[np.ndarray, float]:
    """Read a wav as mono float in ``[-1, 1]``.

    16-bit PCM goes through :mod:`wave` exactly as before -- the arithmetic
    below is unchanged, so every number previously produced from a 16-bit file
    is reproduced bit for bit.

    Anything else (IEEE float32, 24-bit, 32-bit integer) is read with
    :mod:`scipy.io.wavfile`.  This matters more than it looks: 511 of the 809
    annotated ELECOM clips are float32, and the previous version raised
    ``unknown format: 3`` on all of them.  Harnesses that catch per-clip
    exceptions -- which ours do -- then score the readable 37% and report a
    clip count without ever mentioning the rest, so file format silently
    became a selection criterion.  (Fixed 2026-09-05.)
    """
    try:
        with wave.open(str(path)) as w:
            fs = float(w.getframerate())
            n_ch = w.getnchannels()
            width = w.getsampwidth()
            raw = w.readframes(w.getnframes())
        if width == 2:
            x = np.frombuffer(raw, dtype=np.int16).astype(float) / 32768.0
            if n_ch > 1:
                x = x.reshape(-1, n_ch).mean(axis=1)
            return x, fs
    except Exception:
        pass  # not a format `wave` handles; fall through to scipy

    try:
        from scipy.io import wavfile
    except ImportError as exc:  # pragma: no cover
        raise ValueError(
            f"{Path(path).name}: not 16-bit PCM and scipy is unavailable "
            "to read it") from exc

    fs_i, data = wavfile.read(str(path))
    x = np.asarray(data)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.issubdtype(x.dtype, np.integer):
        # scale by the type's positive full scale, as the 16-bit path does
        x = x.astype(float) / float(np.iinfo(x.dtype).max + 1)
    else:
        x = x.astype(float)
    return x, float(fs_i)


def resample_linear(x: np.ndarray, fs: float, target_fs: float) -> np.ndarray:
    """Linear resampling, matching what the ELECOM feature pipeline does."""
    if abs(fs - target_fs) < 1e-9:
        return x
    n_out = int(round(len(x) * target_fs / fs))
    if n_out < 2:
        return x
    return np.interp(np.linspace(0, len(x) - 1, n_out), np.arange(len(x)), x)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def modal_multiple(mult):
    """Plurality multiple over a clip's frames; ties go to the smallest.

    Returns ``(multiple, tied)``.  Smallest-wins is the rule the published
    results use (it is what ``pandas.Series.mode().iloc[0]`` returned before
    pandas was dropped), so the numbers reproduce exactly.
    """
    vals, counts = np.unique(np.asarray(mult, dtype=float), return_counts=True)
    top = vals[counts == counts.max()]
    return float(top.min()), bool(top.size > 1)


def score_tracked_f0(
    times: np.ndarray,
    f0: np.ndarray,
    contours: list | None,
    *,
    gt_version: str = "v2",
    max_gap_s: float = 0.2,
    derived: dict | None = None,
) -> dict | None:
    """Score one clip's tracked F0 against its annotation.

    Parameters
    ----------
    times, f0 : ndarray
        Tracker output; ``f0`` is ``NaN`` on unvoiced frames.
    contours : list
        This clip's entry from :func:`~pitch_tracker.groundtruth.parse_raw_sheet`.
    gt_version : {"v2", "v1"}
        Which ground-truth derivation to score against.
    max_gap_s : float
        Grid points further than this from any voiced tracker frame are not
        scored, so a sparse track is not silently interpolated across silence.
    derived : dict, optional
        An already-derived reference with the keys ``derive_f0_curve_v2``
        returns (``grid_t``, ``f0_derived``, ``c1_interp``, ``n_contours``,
        ``harmonic_number_of_contour1``, optionally ``gt_spread_cents``).  Pass
        it for a corpus that annotates F0 directly; ``contours`` is then
        ignored.

    Returns
    -------
    dict or None
        ``None`` when the annotation yields no usable F0 curve or nothing
        overlaps.
    """
    if derived is None:
        derive = derive_f0_curve_v2 if gt_version == "v2" else derive_f0_curve
        derived = derive(contours)
    if derived is None:
        return None

    grid_t = derived["grid_t"]
    ref_all = derived["f0_derived"]
    ok = np.isfinite(f0)
    if ok.sum() < 2:
        return None

    pred = np.interp(grid_t, times[ok], f0[ok], left=np.nan, right=np.nan)
    gap = np.abs(times[ok][None, :] - grid_t[:, None]).min(axis=1)
    pred[gap > max_gap_s] = np.nan

    mask = np.isfinite(pred) & np.isfinite(ref_all)
    n = int(mask.sum())
    base = {
        "n_scored": n,
        "n_contours": derived["n_contours"],
        "harmonic_number_of_contour1": derived["harmonic_number_of_contour1"],
    }
    if n == 0:
        return base

    pv, rv = pred[mask], ref_all[mask]
    e_strict = cents_error(pv, rv)
    e_oct, mult = best_octave_cents_error(pv, rv)
    mode_mult, mode_tied = modal_multiple(mult)

    mask_c1 = mask & np.isfinite(derived["c1_interp"])
    e_c1 = (cents_error(pred[mask_c1], derived["c1_interp"][mask_c1])
            if mask_c1.sum() else np.array([]))

    spread = derived.get("gt_spread_cents")
    base.update({
        "pred_mean_hz": float(np.mean(pv)),
        "pred_std_hz": float(np.std(pv)),
        "gt_mean_hz": float(np.mean(rv)),
        "gt_std_hz": float(np.std(rv)),
        "level_bias_cents": float(np.median(e_strict)),
        "mae_hz_strict": float(np.mean(np.abs(pv - rv))),
        "rmse_hz_strict": float(np.sqrt(np.mean((pv - rv) ** 2))),
        "medae_cents_strict": float(np.median(np.abs(e_strict))),
        "gross_error_rate_strict_50c": float(np.mean(np.abs(e_strict) > 50)),
        "medae_cents_octave_tolerant": float(np.median(np.abs(e_oct))),
        "gross_error_rate_octave_tolerant_50c": float(np.mean(np.abs(e_oct) > 50)),
        "octave_multiple_mode": mode_mult,
        "octave_mode_tied": mode_tied,
        "frac_multiple_1": float(np.mean(mult == 1)),
        "medae_cents_vs_contour1": float(np.median(np.abs(e_c1))) if len(e_c1) else np.nan,
        "corr_pearson": float(np.corrcoef(pv, rv)[0, 1]) if n >= 3 else np.nan,
        "coverage": float(n / max(int(np.isfinite(ref_all).sum()), 1)),
        "gt_spread_cents": (float(np.nanmedian(spread[mask]))
                            if spread is not None and np.isfinite(spread[mask]).any()
                            else np.nan),
    })
    return base
