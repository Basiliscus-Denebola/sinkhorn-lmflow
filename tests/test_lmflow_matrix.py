import numpy as np
import pytest
from sinkhorn_lmflow.core.linalg import lm_matrix
from sinkhorn_lmflow.core.spectral import standardized_B, nontrivial_sigma_max


@pytest.mark.parametrize("theta", [0, .25, .9, 1])
def test_normalized_spectrum(theta):
    P = np.random.default_rng(9).uniform(.01, 1, (5, 3))
    eps = .3
    r, c = P.sum(1), P.sum(0)
    w = np.sqrt(eps/np.r_[r, c])
    C = w[:, None]*lm_matrix(P, eps, theta)*w[None, :]
    B = standardized_B(P)
    expected = np.block([[np.eye(5), theta*B], [theta*B.T, np.eye(3)]])
    np.testing.assert_allclose(C, expected, atol=1e-14)
    s = np.linalg.svd(B, compute_uv=False)
    np.testing.assert_allclose(s[0], 1, atol=1e-14)
    assert nontrivial_sigma_max(B) == pytest.approx(s[1])
    eig = np.sort(np.r_[1-theta*s, 1+theta*s, np.ones(2)])
    np.testing.assert_allclose(np.linalg.eigvalsh(C), eig, atol=1e-14)
