import tempfile
import unittest
from pathlib import Path

from app.stores.ui_state import UiStateStore


class UiStateStoreMarketDataLabTest(unittest.TestCase):
    def test_market_data_lab_state_normalizes_symbols_and_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = UiStateStore(Path(tmpdir) / "ui_state.json")

            state = store.set_market_data_lab_state(
                {
                    "watchlist_symbols": "aapl, msft, nvda, invalid symbol, aapl",
                    "last_viewed_symbol": "msft",
                    "chart_interval": "5min",
                }
            )

            self.assertEqual(state["watchlist_symbols"], ["AAPL", "MSFT", "NVDA"])
            self.assertEqual(state["last_viewed_symbol"], "MSFT")
            self.assertEqual(state["chart_interval"], "5min")

    def test_market_data_lab_state_falls_back_to_first_symbol_when_selected_removed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = UiStateStore(Path(tmpdir) / "ui_state.json")

            state = store.set_market_data_lab_state(
                {
                    "watchlist_symbols": "AAPL,MSFT",
                    "last_viewed_symbol": "NVDA",
                    "chart_interval": "bad",
                }
            )

            self.assertEqual(state["watchlist_symbols"], ["AAPL", "MSFT"])
            self.assertEqual(state["last_viewed_symbol"], "AAPL")
            self.assertEqual(state["chart_interval"], "1day")

    def test_market_data_lab_onboarding_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = UiStateStore(Path(tmpdir) / "ui_state.json")

            initial = store.get_market_data_lab_onboarding()
            hidden = store.set_market_data_lab_onboarding(True)
            shown = store.set_market_data_lab_onboarding(False)

            self.assertEqual(initial, {"dismissed": False, "enabled": True})
            self.assertEqual(hidden, {"dismissed": True, "enabled": False})
            self.assertEqual(shown, {"dismissed": False, "enabled": True})


if __name__ == "__main__":
    unittest.main()
