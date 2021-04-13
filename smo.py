import numpy as np


def smo(X, y, C, tol):
    m = X.shape[0]
    alpha_vector = np.zeros(m)
    bias = 0

    # Hypothesis on an 'x' training example
    def hypothesis(X, y, alpha_vector, bias, vectorized=False, x=None):
        if not vectorized:
            scalars = y * alpha_vector
            w = X * scalars[:, np.newaxis]

            f = w @ x

            return f.sum() + bias
        else:
            scalars = y * alpha_vector
            w = X * scalars[:, np.newaxis]
            f = (w @ X.T).T

            return f.sum(axis=1) + bias

    E_vector = hypothesis(X, y, alpha_vector, bias, vectorized=True) - y

    def examine_example(X, y, i):
        yi = y[i]
        alpha2 = alpha_vector[i]
        ei = hypothesis(X, y, alpha_vector, bias, X[i]) - y[i]
        ri = ei * y[i]

        if (ri < -tol and alpha2 < C) or (ri > tol and alpha2 > 0):
            if alpha_vector[(alpha_vector > 0) & (alpha_vector < C)].size > 1:
                pass

    def take_step(X, y, i, j):
        if i == j:
            return 0

        alpha1 = alpha_vector[i]
        alpha2 = alpha_vector[j]
        yi = y[i]
        ei = hypothesis(X, y, alpha_vector, bias, X[i]) - y[i]

        s = y[i] * y[j]

        L = max(0, alpha2)


    def kernel(i, j):
        pass

    num_changed = 0
    examine_all = True

    while num_changed > 0 or examine_all:
        num_changed = 0;
        if examine_all:
            pass
        else:
            pass

        if examine_all == 1:
            examine_all = False
        elif num_changed == 0:
            examine_all = True
