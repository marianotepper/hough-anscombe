import numpy as np


def compute(arr, sigma, alpha=1):
    v = np.maximum((arr / alpha) + (3. / 8.) + (sigma / alpha) ** 2, 0)
    f = 2. * np.sqrt(v)
    return f
