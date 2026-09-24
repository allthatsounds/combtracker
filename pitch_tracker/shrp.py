"""phase._shrp — Subharmonic-to-Harmonic Ratio pitch detection (Sun 2002).

The SHRP algorithm estimates the fundamental frequency *and* a continuous
voice-quality descriptor — the Subharmonic-to-Harmonic Ratio (SHR) — by
exploiting the log-frequency structure of the magnitude spectrum.

For a pure harmonic source the spectrum has lines at ``f₀, 2f₀, 3f₀, …``
For a period-doubled (biphonic / creaky / chaotic) source it additionally
has lines at ``½f₀, ³⁄₂f₀, ⁵⁄₂f₀, …``  The Harmonic Sum and Subharmonic
Sum quantities

.. math::

   \\mathrm{HS}(f) \\;=\\; \\sum_{k=1}^{K} \\log |X(k\\,f)|, \\qquad
   \\mathrm{SS}(f) \\;=\\; \\sum_{k=1}^{K} \\log |X((k - \\tfrac12)\\,f)|

peak at the true fundamental.  SHRP picks
``f̂₀ = argmax_f HS(f)`` and reports ``SHR = SS(f̂₀) − HS(f̂₀)`` in the
log domain (equivalently ``SS/HS`` in the linear domain).  High SHR
indicates substantial subharmonic content, i.e. non-modal phonation.

Reference
---------
X. Sun, "Pitch determination and voice quality analysis using
Subharmonic-to-Harmonic Ratio," Proc. ICASSP 2002, Orlando.

This implementation is a from-scratch numpy port, parameterised for any
quasi-harmonic signal (originally speech; here, infrasonic elephant
rumbles).  It works on a precomputed magnitude spectrogram so the caller
controls the front end (STFT, filterbank, or any other invertible
representation) — see :func:`extract_shrp` for the convenience wrapper
that bundles an internal STFT.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ShrpResult:
    """Result of one SHRP pass.

    Attributes
    ----------
    times : ndarray, shape (T,)
        Frame indices (or time in seconds when produced by
        :func:`extract_shrp`).
    f0 : ndarray, shape (T,)
        Estimated fundamental frequency per frame, in Hz.  ``NaN`` for
        frames where no candidate exceeded the magnitude floor.
    shr : ndarray, shape (T,)
        Subharmonic-to-Harmonic Ratio per frame, dimensionless.  Typical
        values: modal voicing ≲ 0.15, biphonic / creaky ≈ 0.3–0.6,
        chaotic / aperiodic close to 1.  ``NaN`` for unvoiced frames.
    confidence : ndarray, shape (T,)
        Per-frame Harmonic Sum normalised to ``[0, 1]`` — a soft
        voicing indicator.
    """

    times: np.ndarray
    f0: np.ndarray
    shr: np.ndarray
    confidence: np.ndarray


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_shrp(
    signal: np.ndarray,
    fs: float,
    *,
    win_seconds: float = 0.5,
    hop_seconds: float = 0.05,
    f0_min: float = 5.0,
    f0_max: float = 80.0,
    n_f0_candidates: int = 128,
    n_harmonics: int = 6,
    n_subharmonics: int = 4,
    mag_threshold_db: float = -40.0,
    smoothness_median: int = 5,
) -> ShrpResult:
    """SHRP F0 + SHR extraction with an internal STFT front end.

    Parameters
    ----------
    signal : ndarray
        Mono audio at sampling rate ``fs``.
    fs : float
        Sampling rate (Hz).
    win_seconds : float
        STFT window length in seconds.  Long windows give fine
        frequency resolution and are appropriate for stationary
        harmonic signals like rumble bodies; default 0.5 s gives 2 Hz
        spectral resolution at fs = 8 kHz.
    hop_seconds : float
        STFT hop in seconds (default 50 ms).
    f0_min, f0_max : float
        F0 search range in Hz.  Default 5–80 Hz is rumble-tuned.
    n_f0_candidates : int
        Number of log-spaced F0 candidates.
    n_harmonics : int
        Number of harmonics summed into HS (and SS).
    n_subharmonics : int
        Number of (k − 1/2) terms summed into SS.  Capped at
        ``n_harmonics``.
    mag_threshold_db : float
        Spectrogram bins below this threshold (relative to the global
        max) are treated as silent.
    smoothness_median : int
        Width of a median filter applied to F0 and SHR after extraction
        (frames).  Set to ``1`` to disable.

    Returns
    -------
    result : ShrpResult
    """
    signal = np.asarray(signal, dtype=float)
    win_len = max(8, int(round(win_seconds * fs)))
    hop = max(1, int(round(hop_seconds * fs)))

    # STFT magnitude (one-sided)
    mag, freqs, frame_times = _stft_magnitude(signal, fs, win_len, hop)

    res = extract_shrp_from_spectrogram(
        mag,
        freqs,
        f0_min=f0_min,
        f0_max=f0_max,
        n_f0_candidates=n_f0_candidates,
        n_harmonics=n_harmonics,
        n_subharmonics=n_subharmonics,
        mag_threshold_db=mag_threshold_db,
        smoothness_median=smoothness_median,
    )
    res.times = frame_times
    return res


def extract_shrp_from_spectrogram(
    magnitude: np.ndarray,
    freqs_hz: np.ndarray,
    *,
    f0_min: float = 5.0,
    f0_max: float = 80.0,
    n_f0_candidates: int = 128,
    n_harmonics: int = 6,
    n_subharmonics: int = 4,
    mag_threshold_db: float = -40.0,
    smoothness_median: int = 5,
) -> ShrpResult:
    """Lower-level entry point that operates on a precomputed spectrogram.

    Parameters
    ----------
    magnitude : ndarray, shape (F, T)
        Non-negative magnitude (or magnitude-squared) spectrogram.
    freqs_hz : ndarray, shape (F,)
        Frequency in Hz of each row of ``magnitude``.
    f0_min, f0_max, n_f0_candidates, n_harmonics, n_subharmonics,
    mag_threshold_db, smoothness_median :
        Same meaning as :func:`extract_shrp`.

    Returns
    -------
    result : ShrpResult
        ``result.times`` is a frame-index array; convert to seconds
        externally if needed.
    """
    magnitude = np.asarray(magnitude, dtype=float)
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    if magnitude.shape[0] != freqs_hz.shape[0]:
        raise ValueError(
            f"magnitude has {magnitude.shape[0]} freq bins but freqs_hz has "
            f"{freqs_hz.shape[0]} entries"
        )
    F_bins, T = magnitude.shape

    eps = 1e-12
    s_max = float(magnitude.max()) if magnitude.size else 1.0
    floor = s_max * 10.0 ** (mag_threshold_db / 20.0)

    log_mag = np.log(np.maximum(magnitude, floor) + eps)

    # Log-spaced F0 candidates.
    f0_cands = np.exp(
        np.linspace(np.log(f0_min), np.log(f0_max), n_f0_candidates)
    )

    n_sub = min(n_subharmonics, n_harmonics)

    # For each candidate F0, precompute the interpolation indices into
    # ``freqs_hz`` for k = 1..K harmonics and k = 0..n_sub-1 subharmonics.
    # We linearly interpolate so the algorithm is robust to coarse / non-
    # uniform frequency grids.
    sh = np.full((n_f0_candidates, T), -np.inf, dtype=float)
    ss = np.full((n_f0_candidates, T), -np.inf, dtype=float)

    f_max = float(freqs_hz[-1])
    f_min_grid = float(freqs_hz[0])

    for ci, f0 in enumerate(f0_cands):
        # Harmonic sum.
        sh_acc = np.zeros(T, dtype=float)
        sh_valid = 0
        for k in range(1, n_harmonics + 1):
            f_t = k * f0
            if f_t > f_max or f_t < f_min_grid:
                continue
            row = _interp_row(log_mag, freqs_hz, f_t)
            sh_acc += row
            sh_valid += 1
        if sh_valid >= 2:
            sh[ci] = sh_acc

        # Subharmonic sum: terms at (k - 1/2) f0  for k=1..n_sub
        ss_acc = np.zeros(T, dtype=float)
        ss_valid = 0
        for k in range(1, n_sub + 1):
            f_t = (k - 0.5) * f0
            if f_t > f_max or f_t < f_min_grid:
                continue
            row = _interp_row(log_mag, freqs_hz, f_t)
            ss_acc += row
            ss_valid += 1
        if ss_valid >= 2:
            ss[ci] = ss_acc

    # Per-frame F0 selection follows Sun (2002):
    #   1. Find local maxima of HS as candidate F0s.
    #   2. Among those within ``equivalence_margin`` of the global max,
    #      prefer the *highest* F0 — this prevents the classical octave-
    #      down error in which the HS at f0/2 ties with HS at f0.
    #   3. SHR at the chosen F0 is the ratio of SS to HS (linear scale).
    #   4. If SHR at the chosen F0 exceeds ``shr_threshold``, the signal
    #      is subharmonic-dominated and we drop F0 by one octave to
    #      report the true sub-fundamental.
    equivalence_margin = 0.5  # log-units; tunable
    shr_threshold = 0.4
    f0_track = np.full(T, np.nan, dtype=float)
    shr_track = np.full(T, np.nan, dtype=float)
    hs_pick = np.full(T, -np.inf, dtype=float)

    for t in range(T):
        col = sh[:, t]
        finite = np.isfinite(col)
        if not finite.any():
            continue
        # Local maxima of HS along the F0 axis.
        peak_idx = _local_maxima(col)
        if not peak_idx:
            ci_best = int(np.argmax(np.where(finite, col, -np.inf)))
        else:
            best_val = max(col[i] for i in peak_idx)
            tied = [i for i in peak_idx if col[i] >= best_val - equivalence_margin]
            ci_best = max(tied)  # prefer highest F0 candidate
        if not np.isfinite(col[ci_best]):
            continue
        f0_pick = float(f0_cands[ci_best])
        ss_val = ss[ci_best, t]
        hs_val = col[ci_best]
        if not np.isfinite(ss_val):
            shr_val = 0.0
        else:
            shr_val = float(np.clip(np.exp(ss_val - hs_val), 0.0, 5.0))

        # Subharmonic-dominated regime: actual F0 is below the picked one.
        if shr_val >= shr_threshold:
            f0_pick = 0.5 * f0_pick

        f0_track[t] = f0_pick
        hs_pick[t] = hs_val
        shr_track[t] = shr_val

    # Confidence: a per-frame HS score, normalised against the theoretical
    # silent baseline (all bins at the floor).  This guards against the
    # case where the input is completely silent — there hs_pick is constant
    # and an unguarded min-max normalisation would assign confidence=1.
    n_terms = max(n_harmonics, 1)
    log_floor = float(np.log(floor + eps))
    silent_baseline = n_terms * log_floor  # HS if every harmonic is at the floor
    valid_hs = hs_pick[np.isfinite(hs_pick)]
    if valid_hs.size > 0:
        hi = float(np.max(valid_hs))
        span = max(hi - silent_baseline, 1e-6)
        confidence = np.where(
            np.isfinite(hs_pick),
            np.clip((hs_pick - silent_baseline) / span, 0.0, 1.0),
            0.0,
        )
    else:
        confidence = np.zeros(T, dtype=float)

    # Median-filter for temporal smoothness.
    if smoothness_median > 1:
        f0_track = _median_filter_nan(f0_track, smoothness_median)
        shr_track = _median_filter_nan(shr_track, smoothness_median)

    return ShrpResult(
        times=np.arange(T, dtype=int),
        f0=f0_track,
        shr=shr_track,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _local_maxima(x: np.ndarray) -> list[int]:
    """Indices of strict local maxima of a 1-D array (finite values only)."""
    out: list[int] = []
    n = len(x)
    for i in range(n):
        v = x[i]
        if not np.isfinite(v):
            continue
        left_ok = i == 0 or not np.isfinite(x[i - 1]) or x[i - 1] <= v
        right_ok = i == n - 1 or not np.isfinite(x[i + 1]) or x[i + 1] <= v
        if left_ok and right_ok:
            # require at least one strictly-smaller neighbour to count as a peak
            stricter = (
                (i > 0 and np.isfinite(x[i - 1]) and x[i - 1] < v)
                or (i < n - 1 and np.isfinite(x[i + 1]) and x[i + 1] < v)
            )
            if stricter:
                out.append(i)
    return out


def _interp_row(log_mag: np.ndarray, freqs_hz: np.ndarray, f_t: float) -> np.ndarray:
    """Linearly interpolate ``log_mag`` along the frequency axis at ``f_t``.

    Returns a vector of length ``T`` (one value per time frame).
    """
    F_bins, _T = log_mag.shape
    idx = np.searchsorted(freqs_hz, f_t)
    if idx <= 0:
        return log_mag[0]
    if idx >= F_bins:
        return log_mag[-1]
    f_lo, f_hi = freqs_hz[idx - 1], freqs_hz[idx]
    if f_hi == f_lo:
        return log_mag[idx]
    w = (f_t - f_lo) / (f_hi - f_lo)
    return (1.0 - w) * log_mag[idx - 1] + w * log_mag[idx]


def _stft_magnitude(
    signal: np.ndarray, fs: float, win_len: int, hop: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Real STFT magnitude with a Hann window.

    Returns ``(magnitude[F, T], freqs[F], times[T])`` where ``times``
    are the frame centres in seconds.
    """
    n = len(signal)
    if n < win_len:
        # Zero-pad the signal up to win_len so we still get one frame.
        padded = np.zeros(win_len, dtype=float)
        padded[:n] = signal
        signal = padded
        n = win_len

    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(win_len) / win_len)
    n_frames = 1 + (n - win_len) // hop
    F_out = win_len // 2 + 1
    mag = np.empty((F_out, n_frames), dtype=float)
    for t in range(n_frames):
        start = t * hop
        frame = signal[start:start + win_len] * win
        spec = np.fft.rfft(frame)
        mag[:, t] = np.abs(spec)
    freqs = np.fft.rfftfreq(win_len, d=1.0 / fs)
    times = (np.arange(n_frames) * hop + win_len // 2) / fs
    return mag, freqs, times


def _median_filter_nan(x: np.ndarray, width: int) -> np.ndarray:
    """1-D median filter that propagates NaNs sensibly.

    For each output position we take the median of the local window
    over the *finite* values; if all values in the window are NaN we
    return NaN.
    """
    if width <= 1:
        return x
    half = width // 2
    out = np.copy(x)
    n = len(x)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        seg = x[lo:hi]
        finite = seg[np.isfinite(seg)]
        if finite.size == 0:
            out[i] = np.nan
        else:
            out[i] = float(np.median(finite))
    return out
