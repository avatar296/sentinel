"""Rolling-window drift detection using Population Stability Index (PSI).

Compares recent score distributions against a baseline to detect model drift.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class DriftReport:
    mean_shift: float      # current_mean - baseline_mean
    std_shift: float       # current_std - baseline_std
    psi: float             # Population Stability Index
    is_drifted: bool       # PSI > threshold
    sample_size: int       # scores in current window
    baseline_size: int     # scores in baseline
    current_mean: float
    baseline_mean: float


class DriftDetector:
    """Tracks ML score distributions and detects drift via PSI.

    Usage:
        detector = DriftDetector(window_size=1000, n_bins=10, psi_threshold=0.2)
        detector.record(0.12)  # call after each prediction
        report = detector.check()
        if report.is_drifted:
            alert(...)
    """

    def __init__(
        self,
        window_size: int = 1000,
        n_bins: int = 10,
        psi_threshold: float = 0.2,
    ) -> None:
        self.window_size = window_size
        self.n_bins = n_bins
        self.psi_threshold = psi_threshold

        self._baseline: list[float] = []
        self._baseline_frozen = False
        self._current: deque[float] = deque(maxlen=window_size)

        # Cached baseline histogram
        self._baseline_hist: np.ndarray | None = None
        self._bin_edges: np.ndarray | None = None
        self._baseline_mean: float = 0.0
        self._baseline_std: float = 0.0

    def record(self, score: float) -> None:
        """Record a prediction score.

        The first `window_size` scores build the baseline.
        After that, scores go into the rolling comparison window.
        """
        if not self._baseline_frozen:
            self._baseline.append(score)
            if len(self._baseline) >= self.window_size:
                self._freeze_baseline()
        else:
            self._current.append(score)

    def _freeze_baseline(self) -> None:
        """Lock the baseline distribution and compute its histogram."""
        arr = np.array(self._baseline)
        self._baseline_mean = float(arr.mean())
        self._baseline_std = float(arr.std())

        # Create histogram with fixed bin edges
        self._baseline_hist, self._bin_edges = np.histogram(
            arr, bins=self.n_bins, range=(0.0, 1.0),
        )
        # Normalize to proportions, add small epsilon to avoid division by zero
        self._baseline_hist = self._baseline_hist.astype(float)
        self._baseline_hist = (self._baseline_hist + 1e-6) / (self._baseline_hist.sum() + self.n_bins * 1e-6)

        self._baseline_frozen = True

    def check(self) -> DriftReport:
        """Compare current window against baseline. Returns a DriftReport."""
        if not self._baseline_frozen:
            return DriftReport(
                mean_shift=0.0,
                std_shift=0.0,
                psi=0.0,
                is_drifted=False,
                sample_size=len(self._baseline),
                baseline_size=len(self._baseline),
                current_mean=0.0,
                baseline_mean=0.0,
            )

        if len(self._current) < 10:
            return DriftReport(
                mean_shift=0.0,
                std_shift=0.0,
                psi=0.0,
                is_drifted=False,
                sample_size=len(self._current),
                baseline_size=len(self._baseline),
                current_mean=0.0,
                baseline_mean=self._baseline_mean,
            )

        current_arr = np.array(self._current)
        current_mean = float(current_arr.mean())
        current_std = float(current_arr.std())

        # Compute current histogram with same bin edges
        current_hist, _ = np.histogram(current_arr, bins=self._bin_edges)
        current_hist = current_hist.astype(float)
        current_hist = (current_hist + 1e-6) / (current_hist.sum() + self.n_bins * 1e-6)

        # PSI = Σ (current - baseline) * ln(current / baseline)
        psi = float(np.sum(
            (current_hist - self._baseline_hist) * np.log(current_hist / self._baseline_hist)
        ))

        return DriftReport(
            mean_shift=current_mean - self._baseline_mean,
            std_shift=current_std - self._baseline_std,
            psi=psi,
            is_drifted=psi > self.psi_threshold,
            sample_size=len(self._current),
            baseline_size=len(self._baseline),
            current_mean=current_mean,
            baseline_mean=self._baseline_mean,
        )

    @property
    def is_ready(self) -> bool:
        """True if baseline is frozen and enough current data exists."""
        return self._baseline_frozen and len(self._current) >= 10
