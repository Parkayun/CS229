import numpy as np
from src.poisson import util
import matplotlib.pyplot as plt
import os


def main(lr, train_path, eval_path, save_path):
    """Problem: Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        save_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    # Run on the validation set, and use np.savetxt to save outputs to save_path as a 1D numpy array
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    clf = PoissonRegression(step_size=lr, verbose=False)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_eval)
    np.savetxt(save_path, pred)

    plot_path = save_path.replace('.txt', '.png')
    plt.figure()
    plt.scatter(y_eval, pred)
    plt.xlabel('true count')
    plt.ylabel('predicted expected count')
    plt.savefig(plot_path)
    # *** END CODE HERE ***


class PoissonRegression:
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=1e-5, max_iter=10000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.
        Update the parameter by step_size * (sum of the gradient over examples)

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        # Initialize self.theta to zero vector if it is None
        # Implement gradient ascent loop with convergence check
        if self.theta is None:
            self.theta = np.zeros(x.shape[1])

        for i in range(self.max_iter):
            old_theta = self.theta.copy()
            pred = np.exp(np.clip(x @ self.theta, -20, 20))
            grad = x.T @ (y - pred) / x.shape[0]
            self.theta += self.step_size * grad

            if np.linalg.norm(self.theta - old_theta) < self.eps:
                break
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Floating-point prediction for each input, shape (n_examples,).
        """
        # *** START CODE HERE ***
        return np.exp(np.clip(x @ self.theta, -20, 20))
        # *** END CODE HERE ***

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main(lr=1e-5,
        train_path=os.path.join(script_dir, 'train.csv'),
        eval_path=os.path.join(script_dir, 'valid.csv'),
        save_path=os.path.join(script_dir, 'poisson_pred.txt'))
