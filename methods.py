
from tools import *


def windowed_eig_extract(X_series, dim, window, beta=0, return_vals=False):
    N = len(X_series)
    thetas, ws = [], []
    for i in range(window, N, 1):

        # build hankel
        Xhan = build_hankel(X_series[i-window:i], dim)
        X0w = Xhan[:, :-1]
        Xpw = Xhan[:, 1:]

        # solve for bottom row of companion matrix
        Xp1 = Xpw[-1]
        X01 = np.vstack((X0w, np.ones(X0w.shape[1])))
        lam = beta * np.eye(dim+1)
        a = (Xp1 @ X01.T) @ np.linalg.inv((X01 @ X01.T) + lam)
        a, b = a[:-1], a[-1:]
        A = np.eye(dim, k=1)
        A[-1] = a

        # get eig decompositon of A
        w, vl = scipy.linalg.eig(A, left=True, right=False)

        # get top eig vector
        sortorder = np.flip(np.argsort(np.abs(w)))
        w = w[sortorder]
        vl = vl[:, sortorder]
        theta = vl[:, 0]
        theta *= np.sign(theta[-1])

        # save eig vec and vals
        thetas.append(theta)
        ws.append(w)

    # average theta
    theta = np.mean(thetas, axis=0)
    Xhan = build_hankel(X_series, dim)
    P_series = theta.real @ Xhan

    if return_vals:
        return P_series, theta, (np.array(ws), np.array(thetas))

    return P_series, theta


def generalized_eig_extract(X_series, dim, beta=0, return_vals=False):

    # build hankel
    Xhan = build_hankel(X_series, dim)
    X0 = Xhan[:, :-1]
    Xp = Xhan[:, 1:]

    # solve generalized eigenvalue problem
    lam = beta * np.eye(dim)
    X0Xp = X0 @ Xp.T
    X0X0 = X0 @ X0.T
    w, vl = scipy.linalg.eig(X0Xp, (X0X0+lam))

    # get top eig vector
    sortorder = np.flip(np.argsort(np.abs(w)))
    w = w[sortorder]
    vl = vl[:, sortorder]
    theta = vl[:, 0]

    P_series = theta.real @ Xhan

    if return_vals:
        return P_series, theta, (w, vl)

    return P_series, theta


class IdentityModel:

    def train(self, X, y, dim, beta=0):

        theta = np.ones(dim)
        y_hat = theta @ X

        # solve for optimal scale and shift
        a, b = solve_scale_shift_ab(y_hat, y)

        theta = theta * a  # pass a into theta so filter is scaled correctly
        pred = theta @ X + b

        return pred, theta, (a, b)

    def test(self, X, theta, params):
        a, b = params
        return theta @ X + b


class LinearModel:
    def train(self, X, y, dim):

        # add ones for bias term
        Xb = np.vstack((X, np.ones(X.shape[1])))

        # solve the least square problem
        theta = (y @ Xb.T) @ np.linalg.inv((Xb @ Xb.T))

        # split the filter from the bias
        theta, b = theta[:-1], theta[-1:]

        # no need to learn scale parameter a as is baked into linear regression
        y_lin = theta @ X + b

        return y_lin, theta, b

    def test(self, X, theta, params):
        b = params
        return theta @ X + b


class WindowedEigModel:

    def train(self, X, y, dim):

        beta = 1e-8
        N = X.shape[1]

        window = int(dim * 2.0)
        thetas = []
        for i in range(window, N, 1):

            Xhan = X[:, i-window:i-dim]

            X0w = Xhan[:, :-1]
            Xpw = Xhan[:, 1:]

            # solve for bottom row of companion matrix
            Xp1 = Xpw[-1]
            X01 = np.vstack((X0w, np.ones(X0w.shape[1])))
            lam = beta * np.eye(dim+1)
            a = (Xp1 @ X01.T) @ np.linalg.inv((X01 @ X01.T) + lam)
            a, b = a[:-1], a[-1:]
            A = np.eye(dim, k=1)
            A[-1] = a

            # get eig decompositon of A
            w, vl = scipy.linalg.eig(A, left=True, right=False)

            # get top eig vector
            sortorder = np.flip(np.argsort(np.abs(w)))
            w = w[sortorder]
            vl = vl[:, sortorder]
            theta = vl[:, 0]
            theta *= np.sign(theta[-1])

            # save eig vec and vals
            thetas.append(theta)

        # average theta
        theta = np.mean(thetas, axis=0)
        y_hat = theta.real @ X

        # solve for optimal scale and shift
        a, b = solve_scale_shift_ab(y_hat, y)

        # pass a into theta so filter is scaled correctly
        theta = theta * a

        # final prediction
        pred = theta.real @ X + b

        return pred, theta, (a, b)

    def test(self, X, theta, params):
        a, b = params
        return theta.real @ X + b


def solve_scale_shift_ab(P_series, Y_series):
    P_series_1 = np.vstack((P_series, np.ones(len(P_series))))
    ab = (Y_series @ P_series_1.T) @ np.linalg.inv((P_series_1 @ P_series_1.T))
    a, b = ab[0], ab[1]

    return a, b


def solve_scale_shift(P_series, Y_series):
    P_series_1 = np.vstack((P_series, np.ones(len(P_series))))
    ab = (Y_series @ P_series_1.T) @ np.linalg.inv((P_series_1 @ P_series_1.T))
    a, b = ab[0], ab[1]
    P_series = P_series*a+b

    return P_series
