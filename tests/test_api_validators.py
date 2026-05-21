import unittest

from fastapi import HTTPException

from app.utils import (
    change_abs_percent,
    date_key_or_none,
    date_or_none,
    epoch_from_iso8601,
    finite_float_or_none,
    first_finite_float,
    iso_date_or_none,
    percent_change,
    percent_of,
    scaled_ratio,
    utc_datetime_or_none,
)
from app.validation import require_non_negative_float, require_positive_float, require_symbols


class ApiValidatorsTest(unittest.TestCase):
    def test_require_symbols_rejects_too_many_symbols(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            require_symbols("AAPL,MSFT,NVDA", max_count=2)

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.detail, "You can request up to 2 symbols at once.")

    def test_require_positive_float_rejects_non_positive_values(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            require_positive_float(0, detail="quantity must be greater than 0.")

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.detail, "quantity must be greater than 0.")

    def test_require_non_negative_float_accepts_zero(self) -> None:
        self.assertEqual(require_non_negative_float(0, detail="must be >= 0."), 0.0)

    def test_require_non_negative_float_rejects_negative_values(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            require_non_negative_float(-0.1, detail="must be >= 0.")

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.detail, "must be >= 0.")

    def test_finite_float_or_none_applies_bounds(self) -> None:
        self.assertEqual(finite_float_or_none("1.5", minimum=0.0, strict_minimum=True), 1.5)
        self.assertIsNone(finite_float_or_none(0, minimum=0.0, strict_minimum=True))
        self.assertEqual(finite_float_or_none(0, minimum=0.0), 0.0)
        self.assertIsNone(finite_float_or_none("nan"))
        self.assertEqual(first_finite_float("nan", "bad", "12.5"), 12.5)
        self.assertEqual(first_finite_float("-1", "2", minimum=0.0, strict_minimum=True), 2.0)

    def test_ratio_helpers_apply_common_percentage_rules(self) -> None:
        self.assertEqual(scaled_ratio("3", "2"), 1.5)
        self.assertEqual(percent_of("-5", "20"), -25.0)
        self.assertEqual(change_abs_percent("110", "100"), (10.0, 10.0))
        self.assertEqual(percent_change("90", "100"), -10.0)
        self.assertIsNone(percent_change("100", "0"))
        self.assertIsNone(scaled_ratio("1", "0"))

    def test_timestamp_helpers_normalize_common_api_shapes(self) -> None:
        parsed = utc_datetime_or_none("2026-04-03T09:00:00+09:00")

        self.assertEqual(parsed.isoformat() if parsed else None, "2026-04-03T00:00:00+00:00")
        self.assertEqual(utc_datetime_or_none("1").isoformat(), "1970-01-01T00:00:01+00:00")
        self.assertEqual(utc_datetime_or_none("1.0").isoformat(), "1970-01-01T00:00:01+00:00")
        self.assertEqual(epoch_from_iso8601("1970-01-01T00:00:01Z"), 1.0)
        self.assertEqual(iso_date_or_none("2026-04-03 09:30:00"), "2026-04-03")
        self.assertEqual(date_or_none("2026-04-03T09:30:00+09:00").isoformat(), "2026-04-03")
        self.assertEqual(date_key_or_none("2026-04-03T09:30:00Z"), "2026-04-03")
        self.assertEqual(date_key_or_none("provider-specific-date"), "provider-specific-date")


if __name__ == "__main__":
    unittest.main()
