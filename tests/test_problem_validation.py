import numpy as np
import pytest
from sinkhorn_lmflow.problems.instance import OTProblem


@pytest.mark.parametrize("a,b,M,eps", [
    ([-.1, 1.1], [.5, .5], np.ones((2, 2)), .1),
    ([0, 1], [.5, .5], np.ones((2, 2)), .1),
    ([.3, .3], [.5, .5], np.ones((2, 2)), .1),
    ([.5, .5], [.5, .5], np.ones((2, 3)), .1),
    ([.5, .5], [.5, .5], np.ones((2, 2)), 0),
    ([.5, .5], [.5, .5], np.ones((2, 2)), -1),
    ([np.nan, .5], [.5, .5], np.ones((2, 2)), .1),
    ([.5, .5], [.5, .5], np.full((2, 2), np.inf), .1),
    ([[.5, .5]], [.5, .5], np.ones((2, 2)), .1),
])
def test_invalid(a, b, M, eps):
    with pytest.raises(ValueError):
        OTProblem(a, b, M, eps)


def test_negative_cost_and_immutable():
    p = OTProblem([1], [1], [[-5]], .1)
    with pytest.raises(ValueError):
        p.M[0, 0] = 1
