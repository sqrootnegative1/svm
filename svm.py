import numpy as np
from enum import Enum
from svm_cvxopt import svm_cvxopt
from smo import SMO

opt_method = Enum("OptMethod", "cvxopt smo")


class SVM:
    def __init__(self, C=1, tol=1e-3, optimization_method=opt_method.cvxopt):
        self.C_ = C
        self.tol_ = tol
        self.X = None
        self.y = None
        self.solved = False
        self.w = None
        self.b = None
        self.support_vector_indices = None
        self.optimization_method = optimization_method


    def fit(self, X, y, show=False):
        """
        Fit a classifier to a feature vector X with labels y

        Returns:
            w = vector orthogonal to separating hyperplane
            b = intercept term
        """

        self.X = X
        self.y = y

        if self.optimization_method == opt_method.cvxopt:
            self.w, self.b, self.support_vector_indices = svm_cvxopt(self.X, self.y, self.C_, self.tol_)
            self.solved = True

        elif self.optimization_method == opt_method.smo:
            smo_solver = SMO(C=self.C_, tol=self.tol_, debug=True)
            self.w, self.b, self.support_vector_indices = smo_solver.train(X, y)

        if show:
            print(f"Calculated w: {self.w}")
            print(f"Calculated b: {self.b}")
    
    
    # Infer
    def predict(self, X):
        return np.sign(X @ self.w + self.b)