# Support Vector Machine
SVM implementation without relying on frameworks

# Usage
Create new data and save it in data.csv:
`python main.py -n`

Train a model using data.csv:
`python main.py`

Specify kernel:
`python main.py --kernel rbf`

# Goals
- [x] Optimize using `cvxopt` package
- [x] Optimize using Sequential Minimal Optimization
- [x] Try different kernels (linear, rbf)
- [ ] Try Polynomial kernel

# Tests
- [ ] MNIST

# Issues
- [ ] Unlike Scikit-learn, this implementation finds large number of support vectors