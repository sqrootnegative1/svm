import cvxopt
import numpy as np


# Hide progress
cvxopt.solvers.options["show_progress"] = False


"""
Solve the SVM dual optimization objective using cvxopt
Returns:
    w = vector orthogonal to separating hyperplane
    b = intercept
 """
def svm_cvxopt(X, y, C, tol):
    m = X.shape[0]
    cvxopt.solvers.options["abstol"] = tol
    cvxopt.solvers.options["reltol"] = tol
    cvxopt.solvers.options["feastol"] = tol

    # Dual coefficients summation over i and j => yi * yj * xi * xj
    const_multipliers = (y.reshape(m, 1) @ y.reshape(1, m)) * (X @ X.T)

    # Prepare matrices according to cvxopt API
    P = cvxopt.matrix(const_multipliers)
    q = cvxopt.matrix(-np.ones((m, 1)))   # (m, 1) because cvxopt API uses q.T
    G = cvxopt.matrix(np.vstack((-1 * np.eye(m), np.eye(m))))
    h = cvxopt.matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
    A = cvxopt.matrix(y.reshape(1, m))  # y.T
    b = cvxopt.matrix(np.zeros(1))

    # Solve
    solver = cvxopt.solvers.qp(P, q, G, h, A, b)

    # Extract alpha values
    alphas = np.array(solver["x"]).flatten()

    # Calculate w using alphas
    w = (alphas * y).T @ X

    # Calculate intercept
    alpha_indices = (alphas > tol).flatten()
    b = np.mean(y[alpha_indices] - (X[alpha_indices] @ w))

    # Done
    return w, b, alpha_indices