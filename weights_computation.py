"""
Observation weight computation for IA-BMLR.

Revised weight formulation (Section 2.2 of the paper):

    w_i = (N / K) * (1 + H_i)^gamma / S_{y_i}(gamma)

where S_k(gamma) = sum_{j: y_j=k} (1 + H_j)^gamma is the within-class
sum of raw entropy terms.  This guarantees sum_i w_i = N for any gamma.

Design note
-----------
Because S_k depends on gamma (a PyMC random variable sampled at every
NUTS iteration), the combined weight w_i must be computed INSIDE the PyMC
model as differentiable PyTensor operations.  See _fit_weighted_bmlr() in
model_ia_bmlr.py for the full in-model implementation.

This module retains compute_class_weights() for reporting and diagnostics
(class weight values are still meaningful to inspect), but compute_entropy_weights()
and compute_observation_weights() are deprecated — their logic has moved
inside the model where automatic differentiation through gamma is available.
"""
import numpy as np
import pytensor.tensor as pt


def compute_class_weights(y):
    """
    Compute inverse-frequency class weights: w_class(k) = N / (K * n_k).

    These are precomputed outside the PyMC model because they depend only
    on fixed class counts and do not involve any random variable.  Under the
    revised formulation the n_{y_i} in this expression cancels algebraically
    with the n_{y_i} in the entropy normalization factor, so these weights are
    used here for reporting/diagnostics rather than being passed into the model
    directly.

    Parameters
    ----------
    y : array-like
        Class labels (integer-encoded).

    Returns
    -------
    weights : ndarray, shape (N,)
        Per-observation class weight w_class(y_i).
    class_weight_dict : dict
        Mapping from class index to its class weight value.
    """
    classes, counts = np.unique(y, return_counts=True)
    N = len(y)
    K = len(classes)

    class_weight_dict = {c: N / (K * count) for c, count in zip(classes, counts)}
    weights = np.array([class_weight_dict[yi] for yi in y])

    return weights, class_weight_dict



