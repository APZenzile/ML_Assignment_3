"""
passive.py
----------

Passive learning selection strategy: 
    instances are chosen unformly at random from the unlabeled pool with 
    no use of the model's current knowledge. This is the 
    baseline against which SASLA and uncertainty-based active 
    learning are compared.
"""

import numpy as np


def select(model, X_pool, n_select, task=None, rng=None):
    """
    Randomly selects n_select instance indices from the unlabeled pool.

    model and task are accepted but UNUSED:
        they're here only so this
        function has the same call signature as the active learning
        selectors (sasla.select, uncertainty.select), letting the
        experiment driver call all three strategies interchangeably.
    """
    if rng is None:
        rng = np.random.default_rng()
    n_available = X_pool.shape[0]
    n_select = min(n_select, n_available)
    return rng.choice(n_available, size=n_select, replace=False)