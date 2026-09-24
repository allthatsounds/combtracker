"""Ablations of the comb objective.

:func:`make_scorer` returns a drop-in replacement for
``pitch_tracker.comb_f0.comb_score_frame`` with one thing changed, which
:func:`~pitch_tracker.comb_f0.estimate_f0` picks up because it looks the
function up on the module at call time. Everything else in the pipeline --
the STFT, the peak picking, the Viterbi decoding -- is untouched, so a
variant isolates the objective and nothing else.

    exempt=False        charge the gap penalty from the first harmonic up
                        instead of from the lowest observed peak, i.e. the
                        missing-fundamental exemption removed, which reduces
                        the objective to a two-way mismatch over slots. It is
                        the demo's second arm and the paper's "without the
                        exemption" ablation (94.1 % against 97.0 %).
    relative_sigma=True match tolerance proportional to the candidate F0
                        rather than absolute Hz.

Use it by swapping the module attribute and putting it back afterwards::

    from pitch_tracker import comb_f0, ablation
    original = comb_f0.comb_score_frame
    comb_f0.comb_score_frame = ablation.make_scorer(exempt=False)
    try:
        result = comb_f0.estimate_f0(x, fs)
    finally:
        comb_f0.comb_score_frame = original

This is the scorer the paper's ablation used, lifted out of the harness that
ran it over the corpus; the gap-penalty sweep and the no-Viterbi variant are
reached through ``estimate_f0``'s own ``gap_penalty`` and
``transition_weight`` arguments.
"""
from __future__ import annotations

import numpy as np


def make_scorer(exempt=True, relative_sigma=False):
    """Return a comb_score_frame with the requested ablation applied."""
    def scorer(peak_freqs, peak_weights, f0_cands, *, sigma_hz=None,
               gap_penalty=None, max_harmonic=40):
        # No defaults, for the reason comb_f0.comb_score_frame gives: a default
        # here would silently score a different objective on a direct call.
        # estimate_f0 always passes both.
        if sigma_hz is None or gap_penalty is None:
            raise TypeError("scorer requires sigma_hz and gap_penalty; the "
                            "pipeline values are sigma_hz=0.94/win_seconds "
                            "and gap_penalty=1.0")
        n_c = len(f0_cands)
        scores = np.full(n_c, -np.inf, dtype=float)
        lowest_k = np.zeros(n_c, dtype=int)
        if peak_freqs.size == 0:
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
            sig = (0.03 * f0) if relative_sigma else sigma_hz
            match = np.exp(-((dev / sig) ** 2))
            explained = float(np.sum(w * match))
            if explained <= 0.0:
                continue
            hit = match > 0.5
            if not hit.any():
                continue
            ks = np.unique(k[hit].astype(int))
            k_lo, k_hi = int(ks[0]), int(ks[-1])
            # THE ABLATION: where does the interior start?
            start = k_lo if exempt else 1
            n_interior = max(k_hi - start + 1, 1)
            n_empty = n_interior - len(ks)
            scores[ci] = (explained - gap_penalty * mean_w * n_empty) / total_w
            lowest_k[ci] = k_lo
        return scores, lowest_k
    return scorer
