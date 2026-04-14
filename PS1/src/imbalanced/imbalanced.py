import numpy as np
from src.imbalanced import util
from random import random
import os

### NOTE : You need to complete logreg implementation first!

from src.linearclass.logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'
# Ratio of class 0 to class 1
kappa = 0.1

def main(train_path, validation_path, save_path):
    """Problem: Logistic regression for imbalanced labels.

    Run under the following conditions:
        1. naive logistic regression
        2. upsampling minority class

    Args:
        train_path: Path to CSV file containing training set.
        validation_path: Path to CSV file containing validation set.
        save_path: Path to save predictions.
    """
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_upsampling = save_path.replace(WILDCARD, 'upsampling')

    # *** START CODE HERE ***
    # Part (b): Vanilla logistic regression
    # Make sure to save predicted probabilities to output_path_naive using np.savetxt() as a 1D numpy array
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_val, y_val = util.load_dataset(validation_path, add_intercept=False)

    clf = LogisticRegression(verbose=False)
    clf.fit(x_train, y_train)
    pred_naive = clf.predict(x_val)
    np.savetxt(output_path_naive, pred_naive)
    util.plot(util.add_intercept(x_val), y_val, clf.theta, output_path_naive.replace('.txt', '.png'))

    pred_label = (pred_naive >= 0.5).astype(int)
    tp = np.sum((pred_label == 1) & (y_val == 1))
    tn = np.sum((pred_label == 0) & (y_val == 0))
    fp = np.sum((pred_label == 1) & (y_val == 0))
    fn = np.sum((pred_label == 0) & (y_val == 1))
    a = (tp + tn) / (tp + tn + fp + fn)
    a0 = tn / (tn + fp)
    a1 = tp / (tp + fn)
    print('naive accuracy:', a)
    print('naive balanced accuracy:', 0.5 * (a0 + a1))
    print('naive A0:', a0)
    print('naive A1:', a1)

    # Part (d): Upsampling minority class
    # Make sure to save predicted probabilities to output_path_upsampling using np.savetxt() as a 1D numpy array
    # Repeat minority examples 1 / kappa times
    reps = int(1 / kappa)
    x_pos = x_train[y_train == 1]
    y_pos = y_train[y_train == 1]
    x_neg = x_train[y_train == 0]
    y_neg = y_train[y_train == 0]

    x_up = np.vstack([x_neg, np.repeat(x_pos, reps, axis=0)])
    y_up = np.concatenate([y_neg, np.repeat(y_pos, reps)])

    clf_up = LogisticRegression(verbose=False)
    clf_up.fit(x_up, y_up)
    pred_up = clf_up.predict(x_val)
    np.savetxt(output_path_upsampling, pred_up)
    util.plot(util.add_intercept(x_val), y_val, clf_up.theta, output_path_upsampling.replace('.txt', '.png'))

    pred_label = (pred_up >= 0.5).astype(int)
    tp = np.sum((pred_label == 1) & (y_val == 1))
    tn = np.sum((pred_label == 0) & (y_val == 0))
    fp = np.sum((pred_label == 1) & (y_val == 0))
    fn = np.sum((pred_label == 0) & (y_val == 1))
    a = (tp + tn) / (tp + tn + fp + fn)
    a0 = tn / (tn + fp)
    a1 = tp / (tp + fn)
    print('upsampling accuracy:', a)
    print('upsampling balanced accuracy:', 0.5 * (a0 + a1))
    print('upsampling A0:', a0)
    print('upsampling A1:', a1)
    # *** END CODE HERE

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main(train_path=os.path.join(script_dir, 'train.csv'),
         validation_path=os.path.join(script_dir, 'validation.csv'),
         save_path=os.path.join(script_dir, 'imbalanced_X_pred.txt'))
