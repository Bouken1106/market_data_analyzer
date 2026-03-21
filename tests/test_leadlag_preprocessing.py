import unittest

import pandas as pd

from app.leadlag.data_adapter import HistoricalPointBatch
from app.leadlag.preprocessing import compute_available_rolling_zscores, prepare_leadlag_dataset
from app.leadlag.schemas import LeadLagConfig


def _points(dates, opens, closes):
    return [
        {"t": day, "o": float(open_price), "c": float(close_price)}
        for day, open_price, close_price in zip(dates, opens, closes)
    ]


class LeadLagPreprocessingTest(unittest.TestCase):
    def test_compute_available_rolling_zscores_uses_previous_window_only(self) -> None:
        index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
        returns = pd.DataFrame({"AAA": [0.10, 0.20, 0.30, 0.00]}, index=index)

        zscores = compute_available_rolling_zscores(returns, window=2)

        self.assertTrue(pd.isna(zscores.loc[pd.Timestamp("2024-01-03"), "AAA"]))
        # Previous window for 2024-01-04 is [0.10, 0.20], so z = (0.30 - 0.15) / 0.05 = 3.0
        self.assertAlmostEqual(float(zscores.loc[pd.Timestamp("2024-01-04"), "AAA"]), 3.0)

    def test_prepare_dataset_maps_next_japan_session(self) -> None:
        dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
        batch = HistoricalPointBatch(
            points_by_symbol={
                "USA1": _points(dates, [100, 100, 101, 102, 103, 104], [100, 101, 102, 103, 104, 105]),
                "USA2": _points(dates, [50, 50, 51, 52, 51, 52], [50, 51, 52, 51, 52, 53]),
                "JP1.T": _points(dates, [100, 100, 100, 100, 100, 100], [100, 101, 102, 103, 104, 105]),
                "JP2.T": _points(dates, [100, 101, 102, 103, 104, 105], [100, 102, 101, 104, 103, 106]),
                "JP3.T": _points(dates, [80, 80, 81, 82, 83, 84], [80, 81, 82, 83, 84, 85]),
                "JP4.T": _points(dates, [60, 60, 61, 62, 63, 64], [60, 61, 62, 63, 64, 65]),
            },
            point_counts={},
            failures={},
        )
        config = LeadLagConfig(
            us_symbols=("USA1", "USA2"),
            jp_symbols=("JP1.T", "JP2.T", "JP3.T", "JP4.T"),
            rolling_window_days=2,
            lambda_reg=0.9,
            n_components=2,
            quantile_q=0.25,
            cfull_start="2024-01-01",
            cfull_end="2024-01-03",
            cyclical_symbols=frozenset({"USA1", "JP1.T", "JP3.T"}),
            defensive_symbols=frozenset({"USA2", "JP2.T", "JP4.T"}),
            refresh=False,
            include_backtest=True,
            include_transfer_matrix=False,
            history_years=10,
        )

        prepared = prepare_leadlag_dataset(config, batch)

        self.assertTrue(prepared.candidate_signal_dates)
        first_signal_date = prepared.candidate_signal_dates[0]
        target_date = prepared.next_target_by_signal_date[first_signal_date]
        self.assertGreater(target_date, first_signal_date)


if __name__ == "__main__":
    unittest.main()
