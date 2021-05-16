import numpy as np
from kernels import kernel_options, linear_kernel, rbf_kernel


class SMO:
    def __init__(self, kernel, C=1, tol=1e-3, gamma=1, debug=False):
        self.X = None
        self.y = None
        self.w = None
        self.b = None
        self.E_vector_ = None
        self.alpha_vector_ = None
        self.kernelized_values = None
        self.tol_ = tol
        self.C_ = C
        self.kernel = kernel
        self.debug = debug
        self.gamma = gamma


    def examine_example(self, i):
        m = self.X.shape[0]
        alpha_i = self.alpha_vector_[i]
        ei = self.E_vector_[i]
        ri = ei * self.y[i]
        
        if (ri < -self.tol_ and alpha_i < self.C_) or (ri > self.tol_ and alpha_i > 0):
            non_boundary_points = np.argwhere((self.alpha_vector_ != 0) & (self.alpha_vector_ != self.C_))
            non_boundary_points = non_boundary_points.reshape(non_boundary_points.shape[0])

            if non_boundary_points.size > 1:
                E_vector_difference = np.abs(self.E_vector_ - ei)
                j = np.argmax(E_vector_difference)

                if self.take_step(i, j):
                    return 1

            np.random.shuffle(non_boundary_points)
            non_boundary_points_list = non_boundary_points.tolist()
            while len(non_boundary_points_list) > 0:
                j = non_boundary_points_list.pop()
                if self.take_step(i, j):
                    return 1

            all_points = np.arange(0, m, 1)
            np.random.shuffle(all_points)
            all_points_list = all_points.tolist()
            while len(all_points_list) > 0:
                j = all_points_list.pop()
                if self.take_step(i, j):
                    return 1

        return 0


    def take_step(self, i, j):
        if i == j:
            return False

        alpha_i = self.alpha_vector_[i]
        alpha_j = self.alpha_vector_[j]
        yi = self.y[i]
        yj = self.y[j]
        ei = self.E_vector_[i]
        ej = self.E_vector_[j]

        s = self.y[i] * self.y[j]
        if s == -1:
            L = max(0, alpha_j - alpha_i)
            H = min(self.C_, alpha_j - alpha_i + self.C_)
        elif s == 1:
            L = max(0, alpha_i + alpha_j - self.C_)
            H = min(self.C_, alpha_i + alpha_j)

        if L == H:
            return False

        Kii = self.calc_kernel(i, i)
        Kjj = self.calc_kernel(j, j)
        Kij = self.calc_kernel(i, j)

        eta = Kii + Kjj - 2 * Kij

        if eta > 0:
            alpha_j_new = alpha_j + float(yj * (ei - ej)) / eta
            if alpha_j_new < L:
                alpha_j_new = L
            elif alpha_j_new > H:
                alpha_j_new = H
        else:
            print(f"smo: degenerate condition detected")
            s = yi * yj
            fi = yi * (ei + self.b) - alpha_i * Kii - s * alpha_j * Kij
            fj = yj * (ej + self.b) - s * alpha_i * Kij - alpha_j * Kjj
            li = alpha_i + s * (alpha_j - L)
            hi = alpha_i + s * (alpha_j - H)
            psi_l = li * fi + L * fj + (1 / 2) * li**2 * Kii + (1 / 2) * L**2 * Kjj + s * L * li * Kij
            psi_h = hi * fi + H * fj + (1 / 2) * hi**2 * Kii + (1 / 2) * H**2 * Kjj + s * H * hi * Kij

            if psi_l < psi_h + self.tol_:
                alpha_j_new = L
            elif psi_l > psi_h + self.tol_:
                alpha_j_new = H
            else:
                alpha_j_new = alpha_j

        if np.abs(alpha_j - alpha_j_new) < self.tol_ * (alpha_j + alpha_j_new + self.tol_):
            return 0

        alpha_i_new = alpha_i + s * (alpha_j - alpha_j_new)

        b_i = -(ei + yi * (alpha_i_new - alpha_i) * Kii + yj * (alpha_j_new - alpha_j) * Kij) + self.b
        b_j = -(ej + yi * (alpha_i_new - alpha_i) * Kij + yj * (alpha_j_new - alpha_j) * Kjj) + self.b
 
        # Updating bias
        if alpha_i_new > 0 and alpha_i_new < self.C_:
            self.b = b_i
        elif alpha_j_new > 0 and alpha_j_new < self.C_:
            self.b = b_j
        else:
            self.b = (b_i + b_j) / 2

        # Update values
        ## Alpha values
        self.alpha_vector_[i] = alpha_i_new
        self.alpha_vector_[j] = alpha_j_new

        ## E vector
        self.E_vector_ = ((self.alpha_vector_ * self.y) @ self.kernelized_values + self.b) - self.y
       
        return 1


    def calc_kernel(self, i=None, j=None):
        """
        The kernel function phi(x)
        Options:
            - Linear kernel : dot product between two vectors
        Returns:
            - Kij if i and j parameters are specified
            - j-th column of kernel matrix if only j is specified
            - i-th row of kernel matrix if only i is specified
            - the whole kernel matrix if no parameter is specified
        """

        if i != None and j != None:
            # single value requested
            # !! Might return nan, make sure to run the kernel function
            # at least without any arguments
            return self.kernelized_values[i][j]

        if j != None:
            # return the j-th column of the kernel matrix
            # !! Might return nan, make sure to run the kernel function
            # at least without any arguments
            return self.kernelized_values[:, j]

        if i != None:
            # return the i-th row of the kernel matrix
            # !! Might return nan, make sure to run the kernel function
            # at least without any arguments
            return self.kernelized_values[i, :]

        # Calculate all kernel values
        if self.kernel == kernel_options.linear:
            kernelized_values = linear_kernel(self.X)
        elif self.kernel == kernel_options.rbf:
            kernelized_values = rbf_kernel(self.X, gamma=self.gamma)
        else:
            raise ValueError("smo: asked to use unknown kernel")

        return kernelized_values


    """
    Fit a classifier to a feature vector X with labels y

    Returns:
        w = vector orthogonal to separating hyperplane
        b = intercept term
    """
    def train(self, X, y):
        m, n = X.shape
        self.X = X
        self.y = y
        self.alpha_vector_ = np.zeros(m, dtype=np.float64)
        self.E_vector_ = np.array(-y, dtype=np.float64)  # model currently predicts 0, so error is the y value
        self.w = np.zeros(n)
        self.b = 0

        # Calculate kernel values beforehand
        self.kernelized_values = self.calc_kernel().reshape(m, m)

        num_changed = 0
        examine_all = True
        iters = 0
        
        while num_changed > 0 or examine_all:
            num_changed = 0
            if examine_all:
                for i in range(m):
                    num_changed += self.examine_example(i)
            else:
                indices_to_examine = np.argwhere((self.alpha_vector_ > 0) & (self.alpha_vector_ < self.C_))
                indices_to_examine = indices_to_examine.reshape(indices_to_examine.shape[0])
                    
                for i in indices_to_examine:
                    num_changed += self.examine_example(i)
            iters += 1
                    
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True
        
        if self.debug:
            print(f"smo: took {iters} iterations to converge")
                
        # Calculate support vector indices
        alpha_indices = (self.alpha_vector_ > self.tol_).flatten()
        indices_to_ignore = (self.alpha_vector_ < self.tol_).flatten()
        self.alpha_vector_[indices_to_ignore] = 0

        return self.alpha_vector_, alpha_indices, self.b