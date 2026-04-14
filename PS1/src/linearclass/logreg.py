import numpy as np
from src.linearclass import util
import os

def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    # Plot decision boundary on top of validation set set
    # Use np.savetxt to save predictions on eval set to save_path as a 1D numpy array
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=False)
    clf = LogisticRegression(verbose=False)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_valid)
    np.savetxt(save_path, pred)
    util.plot(util.add_intercept(x_valid), y_valid, clf.theta, save_path.replace('.txt', '.png'))
    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, max_iter=1000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Shape (n_examples, dim).
            y: Shape (n_examples,).
        """
        x = util.add_intercept(x)
        # *** START CODE HERE ***
        if self.theta is None:
            self.theta = np.zeros(x.shape[1])

        for i in range(self.max_iter):
            old_theta = self.theta.copy()
            h = 1 / (1 + np.exp(-(x @ self.theta)))
            grad = x.T @ (h - y) / x.shape[0]
            d = h * (1 - h)
            hessian = (x.T * d) @ x / x.shape[0]
            self.theta -= np.linalg.solve(hessian, grad)

            if np.sum(np.abs(self.theta - old_theta)) < self.eps:
                break
        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        x = util.add_intercept(x)
        # *** START CODE HERE ***
        return 1 / (1 + np.exp(-(x @ self.theta)))
        # *** END CODE HERE ***

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    main(train_path=os.path.join(script_dir, 'ds1_train.csv'),
         valid_path=os.path.join(script_dir, 'ds1_valid.csv'),
         save_path=os.path.join(script_dir, 'logreg_pred_1.txt'))

    main(train_path=os.path.join(script_dir, 'ds2_train.csv'),
         valid_path=os.path.join(script_dir, 'ds2_valid.csv'),
         save_path=os.path.join(script_dir, 'logreg_pred_2.txt'))
