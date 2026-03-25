import unittest

import numpy as np
import pandas as pd

from app.leadlag.evaluation import evaluate_long_short
from app.leadlag.signals import SignalObservation


class LeadLagEvaluationTest(unittest.TestCase):
    def test_recent_1m_summary_uses_signal_date_boundary(self) -> None:
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
        recent = result["recent_1m_summary"]

        self.assertEqual(recent["signal_days"], 2)
        self.assertEqual(recent["range"], {"from": "2024-02-01", "to": "2024-03-01"})
        self.assertEqual(recent["signal_range"], {"from": "2024-02-01", "to": "2024-03-01"})
        self.assertEqual(recent["target_range"], {"from": "2024-02-02", "to": "2024-03-04"})


if __name__ == "__main__":
    unittest.main()
