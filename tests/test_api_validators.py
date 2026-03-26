import unittest

from fastapi import HTTPException

from app.api.validators import require_non_negative_float, require_positive_float, require_symbols
from app.utils import finite_float_or_none


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


if __name__ == "__main__":
    unittest.main()
