import numpy as np
from src.linearclass import util
import os


def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Train a GDA classifier
    # Plot decision boundary on validation set
    # Use np.savetxt to save outputs from validation set to save_path
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=False)
    clf = GDA()
    clf.fit(x_train, y_train)
    pred = clf.predict(x_valid)
    np.savetxt(save_path, pred)
    util.plot(util.add_intercept(x_valid), y_valid, clf.theta, save_path.replace('.txt', '.png'))
    # *** END CODE HERE ***


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, max_iter=10000, eps=1e-5, verbose=True):
        """
        Args:
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            verbose: Print fitted parameters after training.
        """
        self.theta = None
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Shape (n_examples, dim).
            y: Shape (n_examples,).
        """
        # *** START CODE HERE ***
        y = y.astype(int)
        phi = np.mean(y == 1)
        mu0 = np.mean(x[y == 0], axis=0)
        mu1 = np.mean(x[y == 1], axis=0)
        mu = np.where(y[:, None] == 1, mu1, mu0)
        sigma = (x - mu).T @ (x - mu) / x.shape[0]

        theta = np.linalg.solve(sigma, mu1 - mu0)
        theta0 = (
            -0.5 * mu1.T @ np.linalg.solve(sigma, mu1)
            + 0.5 * mu0.T @ np.linalg.solve(sigma, mu0)
            + np.log(phi / (1 - phi))
        )
        self.theta = np.r_[theta0, theta]
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        x = util.add_intercept(x)
        # *** START CODE HERE ***
        return self._sigmoid(x @ self.theta)
        # *** END CODE HERE ***

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))

    main(train_path=os.path.join(script_dir, 'ds1_train.csv'),
         valid_path=os.path.join(script_dir, 'ds1_valid.csv'),
         save_path=os.path.join(script_dir, 'gda_pred_1.txt'))

    main(train_path=os.path.join(script_dir, 'ds2_train.csv'),
         valid_path=os.path.join(script_dir, 'ds2_valid.csv'),
         save_path=os.path.join(script_dir, 'gda_pred_2.txt'))
