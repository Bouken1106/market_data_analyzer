import unittest

import numpy as np
import pandas as pd

from app.leadlag.evaluation import evaluate_long_short
from app.leadlag.signals import SignalObservation


class LeadLagEvaluationTest(unittest.TestCase):
    def test_summary_range_uses_signal_dates_and_target_range_is_separate(self) -> None:
        observations = []
        for signal_date, target_date, realized_top in (
            ("2024-01-31", "2024-02-01", 0.01),
            ("2024-02-01", "2024-02-02", 0.02),
            ("2024-03-01", "2024-03-04", 0.03),
        ):
            observations.append(
                SignalObservation(
                    signal_date=pd.Timestamp(signal_date),
                    target_date=pd.Timestamp(target_date),
                    predicted=pd.Series([2.0, 1.0, -1.0, -2.0], index=["A", "B", "C", "D"], dtype=np.float64),
                    realized=pd.Series([realized_top, 0.0, 0.0, -realized_top], index=["A", "B", "C", "D"], dtype=np.float64),
                    factors=np.asarray([0.0], dtype=np.float64),
                    eigenvalues=np.asarray([1.0], dtype=np.float64),
                    us_loadings=np.zeros((1, 1), dtype=np.float64),
                    jp_loadings=np.zeros((4, 1), dtype=np.float64),
                    transfer_matrix=np.zeros((4, 1), dtype=np.float64),
                )
            )

        result = evaluate_long_short(tuple(observations), quantile_q=0.25)
        summary = result["summary"]

        self.assertAlmostEqual(summary["period_return_pct"], 12.4448, places=4)
        self.assertEqual(summary["signal_days"], 3)
        self.assertEqual(summary["range"], {"from": "2024-01-31", "to": "2024-03-01"})
        self.assertEqual(summary["signal_range"], {"from": "2024-01-31", "to": "2024-03-01"})
        self.assertEqual(summary["target_range"], {"from": "2024-02-01", "to": "2024-03-04"})


if __name__ == "__main__":
    unittest.main()
