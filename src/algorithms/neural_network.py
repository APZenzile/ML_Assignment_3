"""
neural_network.py
------------------

The fundamental algorithm shared by ALL THREE learning strategies:
    a single-hidden-layer feedforward neural network trained with
    stochastic gradient descent (SGD) and L2 weight decay.

Uses scikit-learn's MLPClassifier / MLPRegressor rather backprop implementation
from scratch.
"""

import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor


def build_model(task, hidden_units, alpha=0.0001, learning_rate_init=0.01,
                 activation='relu', random_state=None, max_iter_per_round=200):
    """
    Constructs a single-hidden-layer NN (Neural Network) trained via SGD.

    task            : 'classification' or 'regression'
    hidden_units    : number of units in the single hidden layer
    alpha           : L2 weight decay coefficient (regularization)
    activation      : any activation function (default to 'relu')
    warm_start=True : lets successive calls to .fit() continue training
                       from current weights instead of reinitializing;
                       this is what allows the "growing labeled set"
                       active learning loop to work.
    """
    common_kwargs = dict(
        hidden_layer_sizes=(hidden_units,),
        activation=activation,
        solver='sgd',
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter_per_round,
        warm_start=True,
        random_state=random_state,
    )
    if task == 'classification':
        return MLPClassifier(**common_kwargs)
    elif task == 'regression':
        return MLPRegressor(**common_kwargs)
    else:
        raise ValueError("task must be 'classification' or 'regression'")


def train_round(model, X_labeled, y_labeled):
    """
    Continues training on the current full labeled set and returns it.
    """
    model.fit(X_labeled, y_labeled)
    return model


def evaluate(model, X_test, y_test, task):
    """
    Evaluates the current model on the providedtest set.

    Classification -> accuracy (higher is better)
    Regression     -> negative MSE (higher is better: for consistent
                       "higher = better" comparison convention.)
    """
    if task == 'classification':
        return model.score(X_test, y_test)
    else:
        preds = model.predict(X_test)
        mse = np.mean((preds - y_test) ** 2)
        return -mse