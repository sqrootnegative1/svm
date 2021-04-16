import numpy as np


class SVM:
    def __init__(self, X, y, C=1, tol=1e-3):
        self.X = X
        self.y = y
        self.C = C
        self.tol = tol
        self.alpha_vector = np.zeros(X.shape[0])
        self.bias = 0
        self.E_vector = np.zeros(X.shape[0])

        for i in range(X.shape[0]):
            self.E_vector[i] = self.hypothesis(i) - self.y[i]


    # Hypothesis of i-th training example
    def hypothesis(self, i):
        m = self.X.shape[0]

        multipliers = self.y * self.alpha_vector

        kernel_products = np.zeros(m)
        for j in range(m):
            kernel_products[j] = self.kernel(j, i)
        
        return (multipliers.T @ kernel_products) + self.bias


    def examine_example(self, i):
        m = self.X.shape[0]
        alpha_i = self.alpha_vector[i]
        ei = self.E_vector[i]
        ri = ei * self.y[i]
        
        if (ri < -self.tol and alpha_i < self.C) or (ri > self.tol and alpha_i > 0):
            non_boundary_points = np.argwhere((self.alpha_vector != 0) & (self.alpha_vector != self.C))
            non_boundary_points = non_boundary_points.reshape(non_boundary_points.shape[0])

            if non_boundary_points.size > 1:
                E_vector_difference = np.abs(self.E_vector - ei)
                j = np.argmax(E_vector_difference)

                if self.take_step(i, j):
                    return 1

            np.random.seed(42)
            np.random.shuffle(non_boundary_points)
            non_boundary_points_list = non_boundary_points.tolist()
            while len(non_boundary_points_list) > 0:
                j = non_boundary_points_list.pop()
                if self.take_step(i, j):
                    return 1

            all_points = np.arange(0, m, 1)
            np.random.seed(42)
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

        alpha_i = self.alpha_vector[i]
        alpha_j = self.alpha_vector[j]
        yi = self.y[i]
        yj = self.y[j]
        ei = self.E_vector[i]
        ej = self.E_vector[j]

        s = self.y[i] * self.y[j]
        L = 0
        H = 0
        if s == -1:
            L = max(0, alpha_j - alpha_i)
            H = min(self.C, alpha_j - alpha_i + self.C)
        elif s == 1:
            L = max(0, alpha_i + alpha_j - self.C)
            H = min(self.C, alpha_i + alpha_j)

        if L == H:
            return False

        Kii = self.kernel(i, i)
        Kjj = self.kernel(j, j)
        Kij = self.kernel(i, j)

        eta = Kii + Kjj - 2 * Kij

        if eta > 0:
            alpha_j_new = alpha_j + yj * (ei - ej) / eta
            if alpha_j_new < L:
                alpha_j_new = L
            elif alpha_j_new > H:
                alpha_j_new = H
        else:
            s = yi * yj
            fi = yi * (ei + self.bias) - alpha_i * Kii - s * alpha_j * Kij
            fj = yj * (ej + self.bias) - s * alpha_i * Kij - alpha_j * Kjj
            li = alpha_i + s * (alpha_j - L)
            hi = alpha_i + s * (alpha_j - H)
            psi_l = li * fi + L * fj + (1 / 2) * li**2 * Kii + (1 / 2) * L**2 * Kjj + s * L * li * Kij
            psi_h = hi * fi + H * fj + (1 / 2) * hi**2 * Kii + (1 / 2) * H**2 * Kjj + s * H * hi * Kij

            if psi_l < psi_h + self.tol:
                alpha_j_new = L
            elif psi_l > psi_h + self.tol:
                alpha_j_new = H
            else:
                alpha_j_new = alpha_j

        self.alpha_vector[j] = alpha_j_new
        self.E_vector[j] = self.hypothesis(j) - self.y[j]
        ej = self.E_vector[j]

        if abs(alpha_j - alpha_j_new) < self.tol * (alpha_j + alpha_j_new + self.tol):
            return False

        alpha_i_new = alpha_i + s * (alpha_j - alpha_j_new)
        self.alpha_vector[i] = alpha_i_new
        self.E_vector[i] = self.hypothesis(i) - self.y[i]
        ei = self.E_vector[i]

        b_i = (ei + yi * (alpha_i_new - alpha_i) * Kii + yj * (alpha_j_new - alpha_j) * Kij) + self.bias
        b_j = (ej + yi * (alpha_i_new - alpha_i) * Kij + yj * (alpha_j_new - alpha_j) * Kjj) + self.bias
        
        # Updating bias
        # Both bound
        if alpha_i_new > 0 and alpha_i_new < self.C and alpha_j_new > 0 and alpha_j_new < self.C:
            self.bias = (b_i + b_j) / 2
            return True

        if alpha_i_new > 0 and alpha_i_new < self.C:
            self.bias = b_i
            return True

        if alpha_j_new > 0 and alpha_j_new < self.C:
            self.bias = b_j
            return True

        return True


    def kernel(self, i, j):
        return self.X[i].T @ self.X[j]


    def smo(self):
        m = self.X.shape[0]
        num_changed = 0
        examine_all = True
        
        while num_changed > 0 or examine_all:
            num_changed = 0
            if examine_all:
                for i in range(m):
                    num_changed += self.examine_example(i)
            else:
                indices_to_examine = np.argwhere((self.alpha_vector != 0) & (self.alpha_vector != self.C))
                indices_to_examine = indices_to_examine.reshape(indices_to_examine.shape[0])

                for i in indices_to_examine:
                    num_changed += self.examine_example(i)

            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True
            
        return self.alpha_vector, self.bias
