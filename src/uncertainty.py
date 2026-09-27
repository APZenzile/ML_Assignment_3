"""
uncertainty.py
--------------

Uncertainty-based active learning selection strategy (ALUS-inspired).

Classification: 
    selects instances whose top-two predicted class
    probabilities are closest together (margin sampling) which means the model
    is least confident on these.

Regression: 
    there are no class probabilities to use, so uncertainty
    is proxied by DISAGREEMENT (variance) across a small ensemble of
    networks independently trained on the same labeled set. High
    variance across the ensemble means the model's prediction is
    unstable.
"""

import numpy as np


def select(model, X_pool, n_select, task='classification',
           ensemble_models=None):
    """
    model           : current trained model
    ensemble_models : list of independently-trained models, required
                       for the regression case; ignored for classification
    """
    n_available = X_pool.shape[0]
    n_select = min(n_select, n_available)

    if task == 'classification':
        probs = model.predict_proba(X_pool)
        sorted_probs = np.sort(probs, axis=1)
        margin = sorted_probs[:, -1] - sorted_probs[:, -2]
        uncertainty_score = -margin

    elif task == 'regression':
        if ensemble_models is None or len(ensemble_models) < 2:
            raise ValueError(
                "Regression uncertainty sampling requires an ensemble "
                "of at least 2 independently trained models."
            )
        preds = np.stack([m.predict(X_pool) for m in ensemble_models], axis=0)
        uncertainty_score = preds.var(axis=0)
    else:
        raise ValueError("task must be 'classification' or 'regression'")

    top_idx = np.argsort(uncertainty_score)[::-1][:n_select]
    return top_idx