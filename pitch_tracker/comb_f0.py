"""Peak-comb fundamental-frequency estimation for infrasonic harmonic calls.

Motivation
----------
The SHRP-locked tracker previously used for ELECOM fails on hand-annotated
elephant rumbles for a specific, diagnosable reason: **the fundamental is
usually not in the recording**.  Measuring the spectrum of annotated calls at
the ground-truth F0 and its multiples shows cases like::

    gtF0 = 14.6 Hz
    H1:+2.0dB  H2:+0.3dB  H3:+1.8dB  H4:+0.2dB  H5:+5.4dB
    H6:+38.1dB H7:+35.4dB H8:+37.9dB      (dB over the in-band noise median)

i.e. all the energy lives at harmonics 6-8 and nothing at all sits at F0.
Two properties of the old algorithm turn this into a hard failure:

1. SHRP scores a candidate ``f`` by ``HS(f) = sum_k log|X(k f)|`` with missing
   bins floored.  A candidate that is a *multiple* of the true F0 lands all of
   its slots on real energy, while the true F0 spends several slots on the
   floor, so multiples routinely outscore the truth.
2. SHRP then breaks near-ties by ``prefer the highest F0 candidate``, which
   pushes the estimate further up the harmonic ladder, and applies a hard
   ``f0 <- f0/2`` whenever SHR crosses a threshold, which injects octave jumps.

The estimator here is built the other way round.  It does not ask "is there
energy where I predict it"; it asks **"does this candidate explain the peaks
that are actually there, and does it predict lines that are conspicuously
missing between them"**.  That single reformulation removes both the
missing-fundamental penalty and the octave ambiguity:

* a candidate at ``2*F0`` leaves every odd harmonic unexplained, so it loses on
  the *explained* term;
* a candidate at ``F0/2`` explains everything but predicts an empty line
  between every observed pair, so it loses on the *gap* term;
* a candidate at the true ``F0`` explains every peak and predicts empty lines
  only *below* the lowest observed peak — which is exactly the
  missing-fundamental region, and is therefore deliberately not penalised.

The score is then tracked over time with Viterbi on a log-frequency lattice so
that isolated noisy frames cannot produce octave jumps.

This is the same quantity a human annotator uses when they read F0 off a
spectrogram as the *spacing* of the harmonic stack rather than the position of
the lowest visible line.

Front end
---------
Only numpy/scipy are needed here.  The filterbank front end used elsewhere in
the pitch tracker (``audfilters`` / ``filterbankphasegrad``) comes from
``cool_frames``; this module deliberately keeps its own STFT so that F0
estimation stays independent of filterbank design choices.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "CombF0Result",
    "estimate_f0",
    "comb_score_frame",
    "pick_peaks",
]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class CombF0Result:
    """F0 trajectory produced by :func:`estimate_f0`.

    Attributes
    ----------
    times : ndarray, shape (T,)
        Frame centres in seconds.
    f0 : ndarray, shape (T,)
        Fundamental frequency in Hz, ``NaN`` where the frame is unvoiced.
    confidence : ndarray, shape (T,)
        Explained-peak weight of the chosen candidate, normalised to ``[0, 1]``.
    n_peaks : ndarray, shape (T,)
        Number of spectral peaks detected in the frame (diagnostic).
    harmonic_numbers : ndarray, shape (T,)
        Harmonic index of the *lowest* explained peak, i.e. how far up the
        stack the visible energy starts.  1 means the fundamental itself is
        present; values > 1 quantify the missing-fundamental depth.
    """

    times: np.ndarray
    f0: np.ndarray
    confidence: np.ndarray
    n_peaks: np.ndarray = field(default_factory=lambda: np.zeros(0))
    harmonic_numbers: np.ndarray = field(default_factory=lambda: np.zeros(0))


# ---------------------------------------------------------------------------
# Spectral peak picking
# ---------------------------------------------------------------------------


def _local_noise_floor(mag: np.ndarray, width_bins: int) -> np.ndarray:
    """Running-median noise floor along the frequency axis.

    A median over a wide window follows the broadband background without being
    dragged up by the narrow harmonic lines themselves.
    """
    n = len(mag)
    if width_bins >= n:
        return np.full(n, float(np.median(mag)))
    half = width_bins // 2
    # Reflect-pad so the floor is defined at the band edges too.
    padded = np.pad(mag, half, mode="reflect")
    strides = np.lib.stride_tricks.sliding_window_view(padded, width_bins)
    return np.median(strides, axis=-1)[:n]


def pick_peaks(
    mag: np.ndarray,
    freqs: np.ndarray,
    *,
    f_lo: float,
    f_hi: float,
    threshold_db: float = 6.0,
    floor_width_hz: float = 60.0,
    max_peaks: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect spectral peaks in ``[f_lo, f_hi]`` with parabolic refinement.

    Returns
    -------
    peak_freqs : ndarray
        Interpolated peak frequencies in Hz, ascending.
    peak_weights : ndarray
        Peak prominence in dB over the local noise floor (>= 0).
    """
    band = (freqs >= f_lo) & (freqs <= f_hi)
    if band.sum() < 5:
        return np.zeros(0), np.zeros(0)
    idx0 = int(np.argmax(band))
    m = mag[band]
    f = freqs[band]

    df = float(f[1] - f[0]) if len(f) > 1 else 1.0
    width = max(5, int(round(floor_width_hz / max(df, 1e-9))))
    floor = _local_noise_floor(m, width)
    eps = 1e-20
    over_db = 20.0 * np.log10((m + eps) / (floor + eps))

    # Strict local maxima above threshold, excluding the two edge bins so that
    # parabolic interpolation always has both neighbours available.
    cand = np.where(
        (over_db[1:-1] >= threshold_db)
        & (m[1:-1] > m[:-2])
        & (m[1:-1] >= m[2:])
    )[0] + 1
    if cand.size == 0:
        return np.zeros(0), np.zeros(0)

    # Parabolic interpolation in the log-magnitude domain.
    lm = np.log(m + eps)
    a, b, c = lm[cand - 1], lm[cand], lm[cand + 1]
    denom = a - 2.0 * b + c
    delta = np.where(np.abs(denom) > 1e-12, 0.5 * (a - c) / np.where(np.abs(denom) > 1e-12, denom, 1.0), 0.0)
    delta = np.clip(delta, -0.5, 0.5)
    pf = f[cand] + delta * df
    pw = over_db[cand]

    if pf.size > max_peaks:
        keep = np.argsort(pw)[-max_peaks:]
        keep.sort()
        pf, pw = pf[keep], pw[keep]

    order = np.argsort(pf)
    del idx0
    return pf[order], pw[order]


# ---------------------------------------------------------------------------
# Comb scoring
# ---------------------------------------------------------------------------


def comb_score_frame(
    peak_freqs: np.ndarray,
    peak_weights: np.ndarray,
    f0_cands: np.ndarray,
    *,
    sigma_hz: float | None = None,
    gap_penalty: float | None = None,
    max_harmonic: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """Score every F0 candidate against one frame's peak set.

    For candidate ``f`` and observed peak ``p`` with weight ``w``:

    * ``k = round(p / f)`` is the harmonic index the candidate assigns to it,
      and ``dev = |p - k f|`` the mismatch **in Hz**;
    * the peak contributes ``w * exp(-(dev / sigma_hz)^2)`` to the
      **explained** term.

    The tolerance is absolute rather than a fraction of ``f`` because the
    physical uncertainty on a peak's location is set by the analysis
    resolution, not by the candidate.  This also makes the score sharply
    sensitive to F0 precision: an F0 that is 1% low places its 12th harmonic
    about 2 Hz away from the observed line, so high harmonics (which is where
    the energy actually lives in these calls) pin F0 down far more tightly
    than the fundamental region ever could.  (The paper quotes this same
    worked example; keep the two in step if either changes.)

    ``sigma_hz`` and ``gap_penalty`` are REQUIRED.  They used to default to
    1.5 Hz and 0.55, and :func:`estimate_f0` always overrode them with
    ``0.94 / T_win`` and 1.0 -- so those defaults were dead for the pipeline
    but live for anyone calling this function directly, who would then have
    silently scored a *different objective* from the one the paper reports.

    The **gap** term counts comb lines ``k f`` for ``k`` strictly between the
    lowest and highest explained harmonic index that no peak falls on, and
    subtracts ``gap_penalty * mean_weight`` for each.  Lines *below* the lowest
    observed peak are never counted, which is what makes the score indifferent
    to a missing fundamental while still rejecting ``f/2``-type candidates.

    The result is normalised by the frame's total peak weight, so scores are
    O(1), comparable across frames of very different loudness, and directly
    commensurable with the Viterbi transition penalty.

    Returns
    -------
    scores : ndarray, shape (n_cands,)
        Comb score per candidate in ``(-inf, 1]``; ``-inf`` where the
        candidate explains nothing.
    lowest_k : ndarray, shape (n_cands,)
        Harmonic index assigned to the lowest explained peak (0 if none).
    """
    if sigma_hz is None or gap_penalty is None:
        missing = ", ".join(
            n for n, v in (("sigma_hz", sigma_hz), ("gap_penalty", gap_penalty))
            if v is None)
        raise TypeError(
            "comb_score_frame requires %s. These previously defaulted to "
            "1.5 Hz and 0.55, which estimate_f0 always replaced with "
            "0.94/T_win and 1.0; a direct call therefore scored a different "
            "objective from the published one. Pass them explicitly -- the "
            "pipeline values are sigma_hz=0.94/win_seconds and "
            "gap_penalty=1.0." % missing)
    # Vectorised over candidates (2026-09-02).  Equivalent to the original
    # per-candidate loop: bit-identical for ascending peaks as ``pick_peaks``
    # returns them (1441 frames, max |delta| = 0), <= 1e-15 for unsorted input
    # (summation order), identical ``lowest_k`` and identical F0 tracks end to
    # end; ~3x faster.
    n_c = len(f0_cands)
    scores = np.full(n_c, -np.inf, dtype=float)
    lowest_k = np.zeros(n_c, dtype=int)
    if peak_freqs.size == 0:
        return scores, lowest_k

    w = np.asarray(peak_weights, dtype=float)
    p = np.asarray(peak_freqs, dtype=float)
    order = np.argsort(p)          # ``k`` is non-decreasing along sorted peaks
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

    # Which harmonic slots are genuinely occupied?
    hit = match > 0.5
    valid = hit.any(axis=1) & (explained > 0.0)
    k_lo = np.where(hit, k, np.inf).min(axis=1)
    k_hi = np.where(hit, k, -np.inf).max(axis=1)
    # Number of distinct occupied slots: because k is non-decreasing along the
    # sorted peaks, a hit duplicates an earlier one iff it equals the running
    # maximum of the hit ks before it.
    k_hit = np.where(hit, k, 0.0)
    prev_max = np.maximum.accumulate(k_hit, axis=1)
    prev_max = np.concatenate([np.zeros((n_c, 1)), prev_max[:, :-1]], axis=1)
    dup = hit & (k == prev_max)
    n_distinct = hit.sum(axis=1) - dup.sum(axis=1)
    n_interior = np.maximum(k_hi - k_lo + 1, 1)
    n_empty = n_interior - n_distinct

    s = (explained - gap_penalty * mean_w * n_empty) / total_w
    scores[valid] = s[valid]
    lowest_k[valid] = k_lo[valid].astype(int)
    return scores, lowest_k

    w = np.asarray(peak_weights, dtype=float)
    p = np.asarray(peak_freqs, dtype=float)
    total_w = float(np.sum(w))
    if total_w <= 0.0:
        return scores, lowest_k
    mean_w = float(np.mean(w))

    for ci, f0 in enumerate(f0_cands):
        k = np.round(p / f0)
        np.clip(k, 1, max_harmonic, out=k)
        dev = np.abs(p - k * f0)
        match = np.exp(-((dev / sigma_hz) ** 2))
        explained = float(np.sum(w * match))
        if explained <= 0.0:
            continue

        # Which harmonic slots are genuinely occupied?
        hit = match > 0.5
        if not hit.any():
            continue
        ks = np.unique(k[hit].astype(int))
        k_lo, k_hi = int(ks[0]), int(ks[-1])
        n_interior = max(k_hi - k_lo + 1, 1)
        n_empty = n_interior - len(ks)

        scores[ci] = (explained - gap_penalty * mean_w * n_empty) / total_w
        lowest_k[ci] = k_lo

    return scores, lowest_k


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _max_plus_linear(prev: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact ``max_j (prev[j] - step * |i - j|)`` in O(n), with argmax.

    The candidate grid is uniform in ``log f``, so the Viterbi transition
    penalty is a linear function of index distance.  A max-plus convolution
    with a linear kernel is a distance transform and needs only a forward and
    a backward sweep, which keeps a fine (1200-point) F0 lattice cheap.
    """
    n = len(prev)
    best = prev.astype(float, copy=True)
    arg = np.arange(n, dtype=np.int32)
    for i in range(1, n):
        cand = best[i - 1] - step
        if cand > best[i]:
            best[i] = cand
            arg[i] = arg[i - 1]
    for i in range(n - 2, -1, -1):
        cand = best[i + 1] - step
        if cand > best[i]:
            best[i] = cand
            arg[i] = arg[i + 1]
    return best, arg


def _stft(signal: np.ndarray, fs: float, win_len: int, hop: int, n_fft: int):
    n = len(signal)
    if n < win_len:
        padded = np.zeros(win_len, dtype=float)
        padded[:n] = signal
        signal = padded
        n = win_len
    win = np.hanning(win_len + 1)[:win_len]
    n_frames = 1 + (n - win_len) // hop
    out = np.empty((n_fft // 2 + 1, n_frames), dtype=float)
    for t in range(n_frames):
        seg = signal[t * hop: t * hop + win_len] * win
        out[:, t] = np.abs(np.fft.rfft(seg, n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    times = (np.arange(n_frames) * hop + win_len / 2.0) / fs
    return out, freqs, times


def estimate_f0(
    signal: np.ndarray,
    fs: float,
    *,
    f0_min: float = 8.0,
    f0_max: float = 60.0,
    n_f0_candidates: int = 1200,
    win_seconds: float = 1.25,
    min_win_seconds: float = 0.5,
    hop_seconds: float = 0.05,
    zero_pad: int = 4,
    band_lo: float = 8.0,
    band_hi: float = 400.0,
    peak_threshold_db: float = 6.0,
    sigma_hz: float | None = None,
    gap_penalty: float = 1.0,
    transition_weight: float = 12.0,
    min_peaks: int = 2,
    voicing_quantile: float = 0.25,
) -> CombF0Result:
    """Estimate F0(t) by comb-matching detected spectral peaks.

    Parameters
    ----------
    signal : ndarray
        Mono audio.
    fs : float
        Sampling rate in Hz.
    f0_min, f0_max : float
        Search range.  The default 8-60 Hz spans documented elephant-rumble
        fundamentals with margin on both sides; the comb score, not the range,
        is what resolves octave ambiguity.
    n_f0_candidates : int
        Log-spaced candidates across the range.
    win_seconds, hop_seconds : float
        STFT geometry.  A 1.25 s window gives a ~3 Hz Hann main lobe, which
        resolves a 15 Hz harmonic spacing comfortably and pins high harmonics
        tightly enough to fix F0 to a few cents.
    min_win_seconds : float
        The window is shortened towards this value on calls too short to
        support ``win_seconds`` (``min(win_seconds, duration / 2.5)``).  A long
        window costs half a window of coverage at each end, which on a 2 s call
        would discard most of the annotation; adapting keeps short calls
        scoreable without giving up resolution on long ones.
    zero_pad : int
        FFT length multiplier; improves peak-frequency interpolation without
        changing the true resolution.
    band_lo, band_hi : float
        Frequency band searched for harmonic peaks.
    peak_threshold_db : float
        A peak must exceed the local noise floor by this much.
    sigma_hz : float or None
        Absolute comb tolerance in Hz.  ``None`` ties it to the window
        (``0.94 / win_eff``, a little under the half-amplitude half-width of a
        Hann main lobe); that evaluates to 0.75 Hz at the default window and
        widens automatically when the window is shortened.  The coefficient is
        empirical, not derived -- see ``tracker_reproduce/src/sigma_sweep.py``,
        which finds octave accuracy flat over ``0.7--1.2 / win_eff``.
    gap_penalty : float
        Empty-interior-line penalty, see :func:`comb_score_frame`.
    transition_weight : float
        Viterbi penalty on ``|log f0[t] - log f0[t-1]|``.  Frame scores are
        normalised to ``(-inf, 1]``, so this is directly interpretable: at 6.0
        an octave jump must buy back ``6 * ln 2 ~ 4.2`` units of score, which a
        single ambiguous frame cannot do.  These calls hold F0 to ~1 Hz over
        seconds, so strong continuity is physically justified.
    min_peaks : int
        Frames with fewer detected peaks are marked unvoiced.
    voicing_quantile : float
        Frames whose explained weight falls below this quantile of the voiced
        frames' explained weight are marked unvoiced.

    Returns
    -------
    result : CombF0Result
    """
    signal = np.asarray(signal, dtype=float)
    if signal.size == 0:
        z = np.zeros(0)
        return CombF0Result(z, z, z, z, z)

    duration = len(signal) / float(fs)
    win_eff = min(float(win_seconds), max(float(min_win_seconds), duration / 2.5))
    win_len = max(16, int(round(win_eff * fs)))
    hop = max(1, int(round(hop_seconds * fs)))
    n_fft = int(2 ** np.ceil(np.log2(win_len * max(1, zero_pad))))
    if sigma_hz is None:
        sigma_hz = 0.94 / win_eff

    mag, freqs, times = _stft(signal, fs, win_len, hop, n_fft)
    T = mag.shape[1]

    f0_cands = np.exp(np.linspace(np.log(f0_min), np.log(f0_max), n_f0_candidates))
    log_f = np.log(f0_cands)
    d_log = float(log_f[1] - log_f[0]) if n_f0_candidates > 1 else 1.0
    step_cost = transition_weight * d_log

    band_hi_eff = min(band_hi, fs / 2.0 * 0.98)
    scores = np.full((T, n_f0_candidates), -np.inf, dtype=float)
    lowest_k = np.zeros((T, n_f0_candidates), dtype=int)
    n_peaks = np.zeros(T, dtype=int)

    for t in range(T):
        pf, pw = pick_peaks(
            mag[:, t], freqs,
            f_lo=band_lo, f_hi=band_hi_eff,
            threshold_db=peak_threshold_db,
        )
        n_peaks[t] = len(pf)
        if len(pf) < min_peaks:
            continue
        sc, lk = comb_score_frame(
            pf, pw, f0_cands, sigma_hz=sigma_hz, gap_penalty=gap_penalty
        )
        scores[t] = sc
        lowest_k[t] = lk

    f0 = np.full(T, np.nan, dtype=float)
    conf = np.zeros(T, dtype=float)
    harm_no = np.zeros(T, dtype=int)

    voiced_t = np.where(np.any(np.isfinite(scores), axis=1))[0]
    if voiced_t.size == 0:
        return CombF0Result(times, f0, conf, n_peaks, harm_no)

    # ---- Viterbi over the voiced frames only -----------------------------
    # Unvoiced frames break the chain rather than forcing an interpolation
    # through a region with no harmonic evidence.
    segments: list[list[int]] = []
    cur = [int(voiced_t[0])]
    for t in voiced_t[1:]:
        if t == cur[-1] + 1:
            cur.append(int(t))
        else:
            segments.append(cur)
            cur = [int(t)]
    segments.append(cur)

    for seg in segments:
        n_s = len(seg)
        cost = np.full((n_s, n_f0_candidates), -np.inf)
        back = np.full((n_s, n_f0_candidates), -1, dtype=np.int32)
        cost[0] = scores[seg[0]]
        for i in range(1, n_s):
            prev = cost[i - 1]
            if not np.any(np.isfinite(prev)):
                cost[i] = scores[seg[i]]
                continue
            best, best_arg = _max_plus_linear(prev, step_cost)
            valid = np.isfinite(scores[seg[i]])
            cost[i] = np.where(valid, best + scores[seg[i]], -np.inf)
            back[i] = np.where(valid, best_arg, -1)

        ci = int(np.argmax(cost[-1]))
        if not np.isfinite(cost[-1, ci]):
            continue
        for i in range(n_s - 1, -1, -1):
            t = seg[i]
            f0[t] = f0_cands[ci]
            conf[t] = scores[t, ci]
            harm_no[t] = lowest_k[t, ci]
            ci = int(back[i, ci])
            if ci < 0:
                break

    # ---- Voicing decision & confidence normalisation ----------------------
    fin = np.isfinite(f0) & np.isfinite(conf)
    if fin.any():
        vals = conf[fin]
        cut = float(np.quantile(vals, voicing_quantile)) if vals.size > 3 else -np.inf
        drop = fin & (conf < min(cut, 0.0))
        f0[drop] = np.nan
        harm_no[drop] = 0
        fin = np.isfinite(f0)
        if fin.any():
            lo, hi = float(np.min(conf[fin])), float(np.max(conf[fin]))
            span = max(hi - lo, 1e-9)
            conf = np.clip((conf - lo) / span, 0.0, 1.0)
        conf[~fin] = 0.0
    else:
        conf[:] = 0.0

    return CombF0Result(
        times=times, f0=f0, confidence=conf,
        n_peaks=n_peaks, harmonic_numbers=harm_no,
    )
