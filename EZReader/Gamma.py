# This class implements gamma distributions.

import random as r
import numpy.core as np


def nextDouble(mean, stdev):
    Beta: float
    done: bool = False
    xx: float = 0
    a = (2.0 * stdev) - 1.0
    a = np.sqrt(1.0 / a)
    b = stdev - np.log(4.0)
    q = stdev + (1.0 / a)
    t = 4.5
    d = 1.0 + np.log(t)
    Beta = mean / stdev
    while not done:
        u1 = r.random()
        u2 = r.random()
        v = a * np.log(u1 / (1.0001 - u1))
        yy = stdev * np.exp(v)
        z = u1 * u1 * u2
        w = b + (q * v) - yy

        if (w + d - (t * z)) >= 0:
            xx = yy
            done = True
        elif w >= np.log(z):
            xx = yy
            done = True
        else:
            done = False

    return Beta * xx
