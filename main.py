import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
#from sklearn import svm
import seaborn as sns

from svm import SVM, opt_method


def main():
    cmap_light = ListedColormap(["wheat", "azure"])
    points = pd.read_csv("data.csv")
    data = points.to_numpy()
    np.random.shuffle(data)

    X = data[:, 0:2]
    y = data[:, -1]

    svm_clf = SVM(C=1, tol=1e-3, optimization_method=opt_method.smo)     # default cvxopt
    svm_clf.fit(X, y, show=True)

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
    pos_support_vectors = np.logical_and(pos_points, svm_clf.support_vector_indices)
    neg_support_vectors = np.logical_and(neg_points, svm_clf.support_vector_indices)
    sns.scatterplot(x=X[pos_support_vectors, 0], y=X[pos_support_vectors, 1], color="darkorange", edgecolor="darkmagenta", linewidth=1.5)
    sns.scatterplot(x=X[neg_support_vectors, 0], y=X[neg_support_vectors, 1], color="cornflowerblue", edgecolor="darkmagenta", linewidth=1.5)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.show()


"""
def main():
    points = pd.read_csv("data.csv")
    data = points.to_numpy()

    cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
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
            #np.random.seed(42)
            np.random.shuffle(data)
            X = data[:, 0:2]
            y = data[:, -1]

            w, b, cvxalpha = fit(X, y)
            cvx_supportvectors = X[cvxalpha.flatten() > 0]
            print(f"qp found\n {w}\nand\n{b}\n")

            xx = np.linspace(0, 10, 100)
            yy = -(w[0] / w[1]) * xx - (b / w[1])

            ax.plot(xx, yy, linestyle="dashdot", color="tab:blue", label="cvxopt")
            #ax.plot(cvx_supportvectors[:, 0], cvx_supportvectors[:, 1], "pr", label="cvx support vectors")
            fig.canvas.draw()

            classifier = SVM(X, y)
            alpha_vector, bias = classifier.smo()
            smo_w = (X * (alpha_vector * y).reshape(y.shape[0], 1)).sum(axis=0)
            S = (alpha_vector > 1e-3).flatten()
            support_vectors = X[S]

            bias = np.mean(y[S] - np.dot(X[S], smo_w))

            print(f"smo_w found\n {smo_w}\nand\n{bias}")
            xxx = np.linspace(0, 10, 100)
            yyy = -(smo_w[0] / smo_w[1]) * xxx - bias / smo_w[1]

            ax.plot(xxx, yyy, linestyle="dashdot", color="purple", label="smo")
            ax.plot(support_vectors[:, 0], support_vectors[:, 1], "r8", label="support vectors")
            plt.legend(loc="upper right")

            print(f"\nsmo support vectors {support_vectors}")
            fig.canvas.draw()

            # sklearn for comparison
            clf = svm.SVC(kernel="linear")
            clf.fit(X, y)
            sk_alphas = np.zeros(X.shape[0])
            sk_alphas[clf.support_] = clf.dual_coef_
            sk_alphas = sk_alphas * y
            sk_w = (X * (sk_alphas * y).reshape(y.shape[0], 1)).sum(axis=0)
            sk_bias = clf.intercept_
            #xxx = np.linspace(0, 10, 100)
            #yyy = -(sk_w[0] / sk_w[1]) * xxx - sk_bias / sk_w[1]
            #print(sk_w, sk_bias)

            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
            zz = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
            zz = zz.reshape(xx.shape)

            plt.figure(figsize=(8, 6))
            plt.contourf(xx, yy, zz, cmap=cmap_light)
            sns.scatterplot(x=X[:, 0], y=X[:, 1], edgecolor="black")
            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())

            plt.show()

 
    cid = fig.canvas.mpl_connect("button_press_event", lambda event: on_plot_click(event, points))
    cid2 = fig.canvas.mpl_connect("key_press_event", on_plot_keypress)

    plt.show()
"""


if __name__ == "__main__":
    main()
