import numpy as np

from sentinel.services.drift_detector import DriftDetector


class TestDriftDetector:
    def test_not_ready_before_baseline(self):
        det = DriftDetector(window_size=100)
        for _ in range(50):
            det.record(0.1)
        report = det.check()
        assert not report.is_drifted
        assert report.psi == 0.0
        assert not det.is_ready

    def test_baseline_freezes_at_window_size(self):
        det = DriftDetector(window_size=100)
        for _ in range(100):
            det.record(0.1)
        assert det._baseline_frozen
        assert det._baseline_hist is not None

    def test_stable_distribution_no_drift(self):
        det = DriftDetector(window_size=100, psi_threshold=0.2)
        rng = np.random.default_rng(42)

        # Baseline: scores around 0.1
        for s in rng.normal(0.1, 0.02, size=100):
            det.record(float(np.clip(s, 0, 1)))

        # Current: same distribution
        for s in rng.normal(0.1, 0.02, size=100):
            det.record(float(np.clip(s, 0, 1)))

        report = det.check()
        assert not report.is_drifted
        assert report.psi < 0.2
        assert report.sample_size == 100

    def test_shifted_distribution_triggers_drift(self):
        det = DriftDetector(window_size=200, psi_threshold=0.2)
        rng = np.random.default_rng(42)

        # Baseline: low scores
        for s in rng.normal(0.1, 0.02, size=200):
            det.record(float(np.clip(s, 0, 1)))

        # Current: much higher scores — clear drift
        for s in rng.normal(0.8, 0.05, size=200):
            det.record(float(np.clip(s, 0, 1)))

        report = det.check()
        assert report.is_drifted
        assert report.psi > 0.2
        assert report.mean_shift > 0.5

    def test_report_fields(self):
        det = DriftDetector(window_size=50)
        rng = np.random.default_rng(0)

        for s in rng.uniform(0, 0.3, size=50):
            det.record(float(s))
        for s in rng.uniform(0, 0.3, size=50):
            det.record(float(s))

        report = det.check()
        assert report.baseline_size == 50
        assert report.sample_size == 50
        assert report.baseline_mean > 0
        assert report.current_mean > 0

    def test_insufficient_current_data(self):
        det = DriftDetector(window_size=50)
        for _ in range(50):
            det.record(0.1)
        # Only 5 current scores — not enough
        for _ in range(5):
            det.record(0.9)
        report = det.check()
        assert not report.is_drifted  # not enough data to judge
