import pytest
import numpy as np


def test_gauss():
    from gauss_filter import gauss_filter
    assert np.allclose(gauss_filter()[2, 3], 0.8106843242768653)
    assert np.allclose(gauss_filter(sigma=2)[-1, -1], 0.7788007830714049)
    assert np.allclose(gauss_filter(mu=2)[6, 7], 0.4008734628486982)
    assert np.allclose(gauss_filter(mu=3, sigma=5)[5, 2], 0.8883114848764877)
