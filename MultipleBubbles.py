import numpy as np
from statsmodels.tsa.stattools import adfuller

# TODO: Add type hinting

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

def bsadfs(m, t, adflag=0, mflag="c"):
    # TODO: determine whether there are extraneous ADF stats being calculated
    qe = [0.9, 0.95, 0.99]
    r0 = 0.01 + 1.8 / np.sqrt(t)
    swindow0 = int(np.floor(r0 * t))
    dim = int(t - swindow0 + 1)

    msadfs = np.zeros((int(m), dim))

    for r2 in range(swindow0, t):
        np.random.seed(123)
        e = np.random.normal(0, 1, (r2, m))
        a = r2**(-1)
        z = e + a
        y = np.apply_along_axis(np.cumsum, 0, z)

        badfs_num_rows: int = int(r2 - swindow0 + 1)
        badfs_vals = np.zeros((badfs_num_rows, m))

        for j in range(m):
            for r1 in range(badfs_num_rows):
                # 1-to-1 replacement of 0.0 with ADF statistic
                badfs_vals[r1][j] = adfuller(y[r1:r2, j], adflag, mflag)[0]

        if r2 == swindow0:
            sadfs = badfs_vals
        else:
            sadfs = np.apply_along_axis(np.max, 0, badfs_vals)

        # Replacing all rows of a column with sadf stat
        msadfs[:, badfs_num_rows] = sadfs

    msadfs_num_cols = np.shape(msadfs)[1]

    # calculating series of statistics for qe quantiles
    quantile_bsadfs = np.zeros((np.shape(qe)[0], msadfs_num_cols))

    for i in range(msadfs_num_cols):
        quantile_bsadfs[:, i] = np.quantile(msadfs[:, i], qe)


    print(msadfs)
    print(quantile_bsadfs)

    return msadfs, quantile_bsadfs
