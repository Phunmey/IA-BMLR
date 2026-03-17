"""
Observation weight computation for IA-BMLR.
"""
import numpy as np


def compute_class_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    N = len(y)
    K = len(classes)

    class_weight_dict = {c: N / (K * count) for c, count in zip(classes, counts)}
    weights = np.array([class_weight_dict[yi] for yi in y])

    return weights, class_weight_dict



