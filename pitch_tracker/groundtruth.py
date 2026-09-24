"""Ground truth from Milos's hand-traced contour annotations.

The annotation workbooks (``merged_contours_{Cohesion,Coordination} 1.xlsx``,
sheet ``RAW``) store one call as pairs of rows: a metadata row carrying
filename / contourID / label followed by the time values in seconds, then an
unlabelled row carrying the matching frequency values in Hz.

Two things about this data drive everything downstream:

1. **contourID = 1 is not the fundamental.**  Across the 784 scored calls its
   median implied harmonic number is ~2, i.e. the annotator traced the lowest
   *visible* harmonic, not F0.  Comparing a tracker against contourID 1
   literally is wrong for roughly half the corpus.
2. **F0 is therefore recovered from the stack, not from any single contour.**
   :func:`derive_f0_curve` takes the median spacing between traced harmonics;
   :func:`derive_f0_curve_v2` (the default) additionally assigns each contour
   an integer harmonic number and averages ``F_i / k_i``, which uses absolute
   positions and divides tracing error by ``k``.

``derive_f0_curve_v2`` also reports ``gt_spread_cents`` — the disagreement
between k-normalised contours, ~19 cents median.  That is the annotation's own
noise floor and no tracker can be expected to beat it.
"""

from __future__ import annotations

import re

import numpy as np
import openpyxl

__all__ = [
    "clip_basename",
    "parse_raw_sheet",
    "derive_f0_curve",
    "derive_f0_curve_v2",
]

_WIN_PATH_RE = re.compile(r"^.*[\\/]")


def clip_basename(raw_filename: str) -> str:
    """'Batch_001_C:\\Posao\\...\\ADDO2012A010.WAV_a0020(1)_15.wav' -> the tail."""
    return _WIN_PATH_RE.sub("", raw_filename)


def parse_raw_sheet(xlsx_path: str) -> dict:
    """{clip_basename: [{"contourID": int, "t": ndarray, "f": ndarray}, ...]}"""
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb["RAW"]
    rows = ws.iter_rows(values_only=True)
    next(rows)  # header
    out: dict = {}
    cur_meta = None
    for r in rows:
        if r[0] is not None:
            cur_meta = (clip_basename(r[0]), r[1], r[2])
            t_vals = np.array([v for v in r[3:] if v is not None], dtype=float)
            pending_t = t_vals
        else:
            f_vals = np.array([v for v in r[3:] if v is not None], dtype=float)
            n = min(len(pending_t), len(f_vals))
            if n >= 2 and cur_meta is not None:
                fname, cid, label = cur_meta
                out.setdefault(fname, []).append(
                    {"contourID": cid, "label": label,
                     "t": pending_t[:n], "f": f_vals[:n]}
                )
    wb.close()
    return out


def derive_f0_curve(contours: list, grid_dt: float = 0.05):
    """Combine a call's traced harmonic stack into one F0(t) curve.

    At each point of a common time grid, interpolate every contour that
    covers that instant, sort the resulting frequencies, take the median
    of consecutive differences as the instantaneous inter-harmonic
    spacing (= F0 estimate). Requires >=2 contours alive at a grid point.
    Also returns, for informational purposes, the harmonic number that
    best matches contourID==1 against the derived F0.
    """
    if not contours:
        return None
    tmin = min(c["t"].min() for c in contours)
    tmax = max(c["t"].max() for c in contours)
    if tmax <= tmin:
        return None
    grid = np.arange(tmin, tmax + grid_dt, grid_dt)
    per_contour_interp = []
    for c in contours:
        t, f = c["t"], c["f"]
        order = np.argsort(t)
        t, f = t[order], f[order]
        vals = np.full(grid.shape, np.nan)
        in_range = (grid >= t.min()) & (grid <= t.max())
        vals[in_range] = np.interp(grid[in_range], t, f)
        per_contour_interp.append(vals)
    stack = np.vstack(per_contour_interp)  # (n_contours, n_grid)

    f0_est = np.full(grid.shape, np.nan)
    for j in range(grid.shape[0]):
        col = stack[:, j]
        col = np.sort(col[np.isfinite(col)])
        if len(col) >= 2:
            diffs = np.diff(col)
            f0_est[j] = float(np.median(diffs))
        elif len(col) == 1:
            # Only one contour alive here: fall back to nothing (can't
            # derive spacing) -- leave NaN, handled by literal comparison.
            pass

    # Which harmonic number is contourID==1?
    c1 = next((c for c in contours if c["contourID"] == 1), contours[0])
    c1_interp = np.interp(grid, c1["t"], c1["f"], left=np.nan, right=np.nan)
    valid = np.isfinite(c1_interp) & np.isfinite(f0_est) & (f0_est > 1e-6)
    harmonic_number = (
        float(np.median(c1_interp[valid] / f0_est[valid])) if valid.any() else np.nan
    )
    return {
        "grid_t": grid, "f0_derived": f0_est, "c1_interp": c1_interp,
        "harmonic_number_of_contour1": harmonic_number,
        "n_contours": len(contours),
    }


def derive_f0_curve_v2(contours: list, grid_dt: float = 0.05):
    """Harmonic-number-aware ground-truth F0 (v2 — the default since 2026-08-24).

    v1 (:func:`derive_f0_curve`) uses the median of consecutive frequency
    differences between traced contours — robust to unknown absolute
    harmonic numbers, but level-limited by hand-tracing noise on
    *differences* and blind to absolute positions.

    v2 assigns each traced contour an integer harmonic number
    ``k_i = round(median(F_i / f0_rough))`` (seeded by v1, refined twice),
    rejects ambiguous contours (|ratio - k| > 0.25), then takes
    ``f0(t) = median_i F_i(t) / k_i``. Tracing error at harmonic k is
    divided by k and absolute positions are used, so both level and
    shape improve; validated against an independent multi-harmonic
    peak-picking instrument (corr 0.08 -> 0.40, |level bias| 25c -> 12c
    on 19 dev clips). Also returns ``gt_spread_cents`` — the internal
    disagreement between k-normalised contours (~19c median), i.e. the
    annotation's own noise floor. Returns None when no contour gets an
    unambiguous harmonic number (typically 2-3-contour clips).
    """
    v1 = derive_f0_curve(contours, grid_dt=grid_dt)
    if v1 is None:
        return None
    grid_t = v1["grid_t"]
    f0_rough = v1["f0_derived"]
    if not np.isfinite(f0_rough).any():
        return None
    rough_med = float(np.nanmedian(f0_rough))
    rough_filled = np.where(np.isfinite(f0_rough), f0_rough, rough_med)

    interp = []
    for c in contours:
        t, f = np.asarray(c["t"], float), np.asarray(c["f"], float)
        ok = np.isfinite(t) & np.isfinite(f)
        if ok.sum() < 2:
            interp.append(np.full_like(grid_t, np.nan))
            continue
        fi = np.interp(grid_t, t[ok], f[ok], left=np.nan, right=np.nan)
        gap = np.array([np.min(np.abs(t[ok] - g)) for g in grid_t])
        fi[gap > 0.15] = np.nan
        interp.append(fi)
    interp = np.array(interp)

    def assign_k(ref):
        ks, keep = [], []
        for fi in interp:
            m = np.isfinite(fi) & np.isfinite(ref)
            if m.sum() < 2:
                ks.append(0); keep.append(False); continue
            ratio = float(np.median(fi[m] / ref[m]))
            k = int(round(ratio))
            ks.append(k)
            keep.append(k >= 1 and abs(ratio - k) <= 0.25)
        return np.array(ks), np.array(keep)

    ks, keep = assign_k(rough_filled)
    if keep.sum() < 1:
        return None
    for _ in range(2):
        norm = np.where(keep[:, None],
                        interp / np.where(ks[:, None] == 0, 1, ks[:, None]),
                        np.nan)
        with np.errstate(all="ignore"):
            f0_fit = np.nanmedian(norm[keep], axis=0)
        fit_med = (float(np.nanmedian(f0_fit))
                   if np.isfinite(f0_fit).any() else rough_med)
        ks, keep = assign_k(np.where(np.isfinite(f0_fit), f0_fit, fit_med))
        if keep.sum() < 1:
            return None

    norm = np.where(keep[:, None],
                    interp / np.where(ks[:, None] == 0, 1, ks[:, None]),
                    np.nan)
    used = norm[keep]
    with np.errstate(all="ignore"):
        f0_fit = np.nanmedian(used, axis=0)
        n_used = np.isfinite(used).sum(axis=0)
        cents_dev = 1200.0 * np.log2(used / f0_fit[None, :])
        spread = np.nanstd(cents_dev, axis=0)
    f0_fit[n_used < 1] = np.nan

    out = dict(v1)
    out["f0_derived"] = f0_fit
    out["f0_derived_v1"] = v1["f0_derived"]
    out["n_used"] = n_used
    out["gt_spread_cents"] = spread
    out["harmonic_ks"] = ks[keep]
    return out
