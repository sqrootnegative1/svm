import numpy as np
import cvxopt


# Return weights and bias for X(feature vector) and y(labels)
def svm_train_dual(X, y, C):
    m = X.shape[0]
    y = y.reshape(-1, 1) * 1.0

    y_X = y * X
    H = np.dot(y_X, y_X.T)

    P = cvxopt.matrix(H)
    q = cvxopt.matrix(-np.ones((m, 1)))
    G = cvxopt.matrix(np.vstack((np.eye(m) * (-1), np.eye(m))))
    h = cvxopt.matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
    A = cvxopt.matrix(y.reshape(1, -1))
    b = cvxopt.matrix(np.zeros(1))

    sol = cvxopt.solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])

    threshold = 1e-2

    w = ((y * alphas).T @ X).reshape(-1, 1)

    S = (alphas > threshold).flatten()

    bias = np.mean(y[S] - np.dot(X[S], w))

    return w, bias


def fit(X, y, C=0.5):
    w, b = svm_train_dual(X, y, C)

    return w, b


def main():
    pass


if __name__ == "__main__":
    main()
