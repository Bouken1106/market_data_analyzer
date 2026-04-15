from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.stores.ui_state import UiStateStore


class UiStateStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)

    def _cache_path(self) -> Path:
        return Path(self._tmpdir.name) / "ui_state.json"

    def test_watchlists_are_persisted_per_namespace(self) -> None:
        store = UiStateStore(cache_path=self._cache_path())
        store.set_symbols(["aapl", "msft"])
        store.set_symbols(["7203.T", "9432.T"], namespace="jp")

        reloaded = UiStateStore(cache_path=self._cache_path())
        persisted = json.loads(self._cache_path().read_text(encoding="utf-8"))

        self.assertEqual(reloaded.get_symbols(), ["AAPL", "MSFT"])
        self.assertEqual(reloaded.get_symbols(namespace="us"), ["AAPL", "MSFT"])
        self.assertEqual(reloaded.get_symbols(namespace="jp"), ["7203.T", "9432.T"])
        self.assertEqual(persisted["watchlists"]["us"], ["AAPL", "MSFT"])
        self.assertEqual(persisted["watchlists"]["jp"], ["7203.T", "9432.T"])


if __name__ == "__main__":
    unittest.main()
