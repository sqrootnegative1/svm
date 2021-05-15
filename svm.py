import numpy as np
from enum import Enum
from svm_cvxopt import svm_cvxopt
from smo import SMO
from kernels import kernel_options, rbf_kernel


opt_method = Enum("OptMethod", "cvxopt smo")


class SVM:
    def __init__(self, C=1, tol=1e-3, optimization_method=opt_method.cvxopt, kernel=kernel_options.linear):
        self.C_ = C
        self.tol_ = tol
        self.X = None
        self.y = None
        self.solved = False
        self.w = None   # only usable in case of a linear classifier
        self.b = None
        self.support_vector_indices = None
        self.alphas = None
        self.optimization_method = optimization_method
        self.kernel = kernel


    def fit(self, X, y, show=False):
        """
        Fit a classifier to training matrix X (m x n) with labels y (m x 1)
        """

        self.X = X
        self.y = y

        if self.optimization_method == opt_method.cvxopt:
            self.alphas, self.support_vector_indices, self.b = svm_cvxopt(self.X, self.y, self.C_, self.tol_)
            self.solved = True

        elif self.optimization_method == opt_method.smo:
            smo_solver = SMO(kernel=self.kernel, C=self.C_, tol=self.tol_, debug=True)
            self.alphas, self.support_vector_indices, self.b = smo_solver.train(X, y)
            self.solved = True

        else:
            raise ValueError("svm: unknown training method provided")

        if self.kernel == kernel_options.linear:
            self.w = (self.alphas * self.y).T @ self.X
            self.w = self.w.reshape(-1, 1)
            self.b = np.mean(self.y[self.support_vector_indices] - (self.X[self.support_vector_indices] @ self.w)) 

        if show:
            print(f"Calculated alpha vector: {self.alphas}")
            print(f"Calculated b: {self.b}")


    def project(self, X_test):
        """
        Project X_test
        """

        # Linear kernel? Use w vector
        if self.kernel == kernel_options.linear:
            return X_test @ self.w + self.b

        # Else project into higher dimensions
        ## RBF
        if self.kernel == kernel_options.rbf:
            return (self.alphas * self.y) @ rbf_kernel(self.X, X_test)
        
        raise ValueError("svm: unknown kernel provided")
    
    
    def predict(self, X):
        """
        Infer trained model on example(s) matrix (or ndarray) X
        """

        return np.sign(self.project(X))