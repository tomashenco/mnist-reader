import numpy as np


def init_weights_and_bias(m1, m2):
    w = np.random.rand(m1, m2)
    b = np.zeros(m2)

    return w.astype(np.float32), b.astype(np.float32)
