"""
sasla.py
--------

Sensitivity Analysis Selective Learning (SASLA)-inspired selection
strategy, based on Prof. Engelbrecht (2001).

Selects instances whose model OUTPUT changes most sharply under a
small perturbation of the INPUT. Such instances lie closest to
decision boundaries (classification) or regions of high curvature
(regression), and are considered most informative to label next.

scikit-learn's MLP does not expose analytic input-output derivatives,
so sensitivity is estimated here via central finite differences, a 
mathematically similar quantity the SASLA paper computes analytically
via the chain rule through the network's weights, just approximated
numerically instead.
"""

import numpy as np


def _model_output(model, X, task):
    """
    Computes per-instance numeric output used for finite-difference sensitivity:
        predicted class-probability vector for classification,
        predicted scalar value for regression.
    """
    if task == 'classification':
        return model.predict_proba(X) 
    else:
        preds = model.predict(X)
        return preds.reshape(-1, 1)

def select(model, X_pool, n_select, task='classification', epsilon=0.001):
    """
    Estimates output sensitivity of each pooled instance to small input
    perturbations (one feature at a time), and selects the n_select
    most sensitive instances.
    """
    n_available, n_features = X_pool.shape
    n_select = min(n_select, n_available)

    sensitivity = np.zeros(n_available)

    for feature_index in range(n_features):
        X_plus = X_pool.copy()
        X_minus = X_pool.copy()
        X_plus[:, feature_index] += epsilon
        X_minus[:, feature_index] -= epsilon

        out_plus = _model_output(model, X_plus, task)
        out_minus = _model_output(model, X_minus, task)

        # central-difference derivative w.r.t. this feature, per output dim
        derivative = (out_plus - out_minus) / (2 * epsilon)

        # accumulates squared derivative magnitude across outputs and features
        sensitivity += np.sum(derivative ** 2, axis=1)

    sensitivity = np.sqrt(sensitivity)

    top_index = np.argsort(sensitivity)[::-1][:n_select]
    return top_index