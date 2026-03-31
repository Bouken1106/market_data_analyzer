import unittest

from app.services.relationship_analysis import build_relationship_analysis


def _build_points(closes: list[float]) -> list[dict[str, float | str]]:
    points: list[dict[str, float | str]] = []
    for index, close in enumerate(closes, start=1):
        points.append(
            {
                "t": f"2024-01-{index:02d}",
                "o": close,
                "h": close,
                "l": close,
                "c": close,
                "v": 1000.0 + index,
            }
        )
    return points


class RelationshipAnalysisTest(unittest.TestCase):
    def test_build_relationship_analysis_returns_matrices_pairs_and_summary(self) -> None:
        payload = build_relationship_analysis(
            {
                "AAPL": _build_points([100, 101, 102, 103, 104, 105, 106, 107]),
                "MSFT": _build_points([200, 202, 204, 206, 208, 210, 212, 214]),
                "XOM": _build_points([90, 91, 89, 90, 88, 89, 87, 88]),
            },
            window_days=5,
            top_pairs=3,
        )

        self.assertEqual(payload["symbols"], ["AAPL", "MSFT", "XOM"])
        self.assertEqual(len(payload["correlation_matrix"]), 3)
        self.assertEqual(len(payload["covariance_matrix"]), 3)
        self.assertEqual(payload["data_summary"]["price_points"], 8)
        self.assertEqual(payload["data_summary"]["return_points"], 7)
        self.assertEqual(payload["pair_candidates"][0]["left"], "AAPL")
        self.assertEqual(payload["pair_candidates"][0]["right"], "MSFT")
        self.assertGreater(payload["pair_candidates"][0]["correlation"], 0.99)
        self.assertIsNotNone(payload["summary"]["average_abs_correlation"])
        self.assertEqual(len(payload["rolling_correlations"]), 3)

    def test_build_relationship_analysis_rejects_insufficient_aligned_data(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            build_relationship_analysis(
                {
                    "AAPL": _build_points([100, 101, 102]),
                    "MSFT": _build_points([200, 201, 202]),
                },
                window_days=3,
            )

        self.assertEqual(str(ctx.exception), "Not enough aligned historical data to analyze relationships.")


if __name__ == "__main__":
    unittest.main()
