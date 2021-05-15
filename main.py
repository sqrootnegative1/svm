import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

from svm import SVM, opt_method
from kernels import kernel_options
from sklearn import svm


def main():
    parser = argparse.ArgumentParser(description="Test Support Vector Machines")
    parser.add_argument("-n", help="register new data", action="store_true")
    parser.add_argument("--kernel", help="specify kernel function (linear, rbf, polynomial). Default linear.", default="linear")
    args = parser.parse_args()

    if args.kernel == "linear":
        kernel_function = kernel_options.linear
    elif args.kernel == "rbf":
        kernel_function = kernel_options.rbf
    elif args.kernel == "polynomial":
        kernel_function = kernel_options.polynomial
    else:
        raise ValueError("Unknown kernel provided")

    if args.n:
        points = []
        fig, ax = plt.subplots()
        ax.set_title("SVM")
        ax.set_xlim([0, 10])
        ax.set_ylim([0, 10])

        def on_plot_click(event, points):
            if event.button == 1 and event.xdata != None and event.ydata != None:
                points.append((event.xdata, event.ydata, -1))
                ax.plot(event.xdata, event.ydata, "g_")
                fig.canvas.draw()
            elif event.button == 3 and event.xdata != None and event.ydata != None:
                points.append((event.xdata, event.ydata, 1))
                ax.plot(event.xdata, event.ydata, "b+")
                fig.canvas.draw()
            elif event.button == 2 and event.xdata != None and event.ydata != None:
                points = points.clear()
                ax.cla()
                ax.set_title("SVM")
                ax.set_xlim([0, 10])
                ax.set_ylim([0, 10])
                fig.canvas.draw()

        def on_plot_keypress(event):
            if event.key == "enter":
                print(f"total points {len(points)}")
                
                data = np.array(points)
                pd.DataFrame(data).to_csv("data.csv", index=None, header=None)
                sys.exit(0)

        fig.canvas.mpl_connect("button_press_event", lambda event: on_plot_click(event, points))
        fig.canvas.mpl_connect("key_press_event", on_plot_keypress)
        plt.show()
    
    else:
        cmap_light = ListedColormap(["wheat", "azure"])
        points = pd.read_csv("data.csv")
        data = points.to_numpy()
        np.random.shuffle(data)

        X = data[:, 0:2]
        y = data[:, -1]

        # Train using custom svm
        svm_clf = SVM(C=1, tol=1e-3, kernel=kernel_function, optimization_method=opt_method.smo)     # default cvxopt
        svm_clf.fit(X, y)

        # Train using scikit-learn
        #svm_clf = svm.SVC(kernel="rbf")
        #svm_clf.fit(X, y)

        # Plot
        pos_points = y > 0
        neg_points = y < 0
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
        zz = svm_clf.predict(np.c_[xx.ravel(), yy.ravel()])
        zz = zz.reshape(xx.shape)
        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, zz, cmap=cmap_light)
        sns.scatterplot(x=X[pos_points, 0], y=X[pos_points, 1], color="darkorange", edgecolor="darkorange")
        sns.scatterplot(x=X[neg_points, 0], y=X[neg_points, 1], color="cornflowerblue", edgecolor="cornflowerblue")

        # For custom svm
        pos_support_vectors = np.logical_and(pos_points, svm_clf.support_vector_indices)
        neg_support_vectors = np.logical_and(neg_points, svm_clf.support_vector_indices)

        # For scikit-learn
        #support_points = np.zeros(y.shape, dtype=np.bool_)
        #support_points[svm_clf.support_] = True
        #pos_support_vectors = np.logical_and(pos_points, support_points)
        #neg_support_vectors = np.logical_and(neg_points, support_points)

        sns.scatterplot(x=X[pos_support_vectors, 0], y=X[pos_support_vectors, 1], color="darkorange", edgecolor="darkmagenta", linewidth=1.5)
        sns.scatterplot(x=X[neg_support_vectors, 0], y=X[neg_support_vectors, 1], color="cornflowerblue", edgecolor="darkmagenta", linewidth=1.5)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())

        plt.show()


if __name__ == "__main__":
    main()
