import numpy as np
from statsmodels.tsa.stattools import adfuller


def gsadf(m, t, adflag=0, mflag="c"):
    qe = [0.9, 0.95, 0.99]
    r0 = 0.01 + 1.8 / np.sqrt(t)
    swindow0 = np.floor(r0 * t)
    dim = t - swindow0 + 1

    y = dgp(t, m)

    gsadf_vals = np.ones((m, 1))

    for j in range(1, m):
        sadfs = np.zeros((int(dim), 1))
        for r2 in range(int(swindow0), t):
            dim0 = r2 - swindow0 + 1
            rwadft = np.zeros((int(dim0), 1))
            for r1 in range(1, int(dim0)):
                selected_array = np.array(y[r1:r2, j])
                adf_test_vals = adfuller(y[r1:r2, j], adflag, mflag)
                rwadft[r1] = adfuller(y[r1:r2, j], adflag, mflag)[0]

            index = int(r2 - swindow0 + 1)
            sadfs[index] = np.max(rwadft)

        gsadf_vals[j] = np.max(sadfs)

    gsadf_critical_vals = np.quantile(gsadf_vals, qe)
    return gsadf_critical_vals


def dgp(n, k):
    np.random.seed(123)
    u0 = 1 / n
    rn = np.random.normal(0, 1, (n, k))
    z = rn + u0
    y = np.apply_along_axis(np.cumsum, 0, z)
    return y

def sadf_gsadf(y, adflag, mflag, info_criterion):
    pass

def badfs():
    pass

def sadf():
    pass

def bsadfs():
    pass
