import numpy as np
import pytest
from sinkhorn_lmflow.problems.instance import OTProblem


@pytest.fixture
def problem():
    rng = np.random.default_rng(123)
    return OTProblem(rng.dirichlet(np.ones(4)), rng.dirichlet(np.ones(3)),
                     rng.uniform(0, 0.3, (4, 3)), 0.2)
