import unittest

import numpy as np

from app.leadlag.subspace_pca import build_prior_subspace, build_target_c0


class LeadLagSubspacePcaTest(unittest.TestCase):
    def test_build_prior_subspace_returns_orthonormal_matrix(self) -> None:
        subspace = build_prior_subspace(
            ("USA1", "USA2"),
            ("JP1.T", "JP2.T", "JP3.T", "JP4.T"),
            cyclical_symbols=frozenset({"USA1", "JP1.T", "JP3.T"}),
            defensive_symbols=frozenset({"USA2", "JP2.T", "JP4.T"}),
        )

        self.assertEqual(subspace.shape, (6, 3))
        gram = subspace.T @ subspace
        np.testing.assert_allclose(gram, np.eye(3), atol=1e-8)

    def test_build_target_c0_has_unit_diagonal(self) -> None:
        cfull = np.array(
            [
                [1.0, 0.3, 0.2, 0.1],
                [0.3, 1.0, 0.4, 0.2],
                [0.2, 0.4, 1.0, 0.5],
                [0.1, 0.2, 0.5, 1.0],
            ],
            dtype=np.float64,
        )
        prior = build_prior_subspace(
            ("USA1", "USA2"),
            ("JP1.T", "JP2.T"),
            cyclical_symbols=frozenset({"USA1", "JP1.T"}),
            defensive_symbols=frozenset({"USA2", "JP2.T"}),
        )

        c0, d0 = build_target_c0(cfull, prior)

        self.assertEqual(d0.shape, (3,))
        np.testing.assert_allclose(np.diag(c0), np.ones(4), atol=1e-8)
        np.testing.assert_allclose(c0, c0.T, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
