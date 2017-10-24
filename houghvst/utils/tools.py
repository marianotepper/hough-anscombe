import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment as \
    _linear_assignment_
from houghvst.utils.stats import half_sample_mode


def linear_assignment(spatial_comp1, spatial_comp2):
    mat = spatial_comp1.T.dot(spatial_comp2)
    mat /= np.linalg.norm(spatial_comp1, axis=0)[:, np.newaxis]
    mat /= np.linalg.norm(spatial_comp2, axis=0)[np.newaxis, :]
    mat = mat.max() - mat

    return _linear_assignment_(mat)[:, 1]


def switch_component_order(A, C, order):
    return A[:, order], C[order, :]


def detrend_and_normalize(tr, detrend_scale, quantile=8):
    if detrend_scale is not None:
        window = int(len(tr) * detrend_scale)
        baseline = [np.percentile(tr[i:i + window], quantile)
                    for i in range(1, len(tr) - window)]
        missing = np.percentile(tr[-window:], quantile)
        missing = np.repeat(missing, window + 1)
        baseline = np.concatenate((baseline, missing))
        tr -= baseline

    hsm = half_sample_mode(tr)
    return (tr - hsm) / (tr.max() - hsm)
