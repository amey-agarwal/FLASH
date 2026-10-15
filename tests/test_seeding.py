import random

import numpy as np
import torch

from fedastro.seeding import seed_everything


def _draw(seed):
    rng = seed_everything(seed)
    return random.random(), np.random.rand(), rng.normal(), torch.randn(3).tolist()


def test_same_seed_same_draws():
    assert _draw(3) == _draw(3)


def test_different_seed_different_draws():
    assert _draw(3) != _draw(4)
