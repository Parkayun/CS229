"""Starter code for the BERT-based spam classification question."""

import numpy as np

import logreg
import util


LEARNING_RATES = [0.01, 0.001, 0.0001, 0.00001, 0.000001]


def compute_accuracy(train_matrix, train_labels, eval_matrix, eval_labels, learning_rate):
    """Train logistic regression with the given learning rate and evaluate it."""
    predictions = logreg.train_and_predict_logreg(
        train_matrix, train_labels, eval_matrix, learning_rate)
    return np.mean(predictions == eval_labels)


def compute_best_logreg_learning_rate(train_matrix, train_labels, val_matrix, val_labels,
                                      learning_rates_to_consider):
    """Compute the best logistic regression learning rate.

    You should only consider learning rates within the learning_rates_to_consider list.
    You should use validation set accuracy as a metric for comparing the different
    learning rates.

    Args:
        train_matrix: The BERT features for the training data.
        train_labels: The spam or not spam labels for the training data.
        val_matrix: The BERT features for the validation data.
        val_labels: The spam or not spam labels for the validation data.
        learning_rates_to_consider: The learning rates to consider.

    Returns:
        The best logistic regression learning rate which maximizes validation set
        accuracy.
    """
    # *** START CODE HERE ***
    best_lr = learning_rates_to_consider[0]
    best_acc = 0
    for lr in learning_rates_to_consider:
        val_acc = compute_accuracy(train_matrix, train_labels, val_matrix, val_labels, lr)
        print('trying lr = %g, val acc = %f' % (lr, val_acc))
        if val_acc > best_acc:
            best_acc = val_acc
            best_lr = lr
    return best_lr
    # *** END CODE HERE ***


def main():
    _, train_labels = util.load_spam_dataset('spam_train.tsv')
    _, val_labels = util.load_spam_dataset('spam_val.tsv')
    _, test_labels = util.load_spam_dataset('spam_test.tsv')

    train_matrix = util.load_bert_encoding('bert_train_matrix.npz')
    val_matrix = util.load_bert_encoding('bert_val_matrix.npz')
    test_matrix = util.load_bert_encoding('bert_test_matrix.npz')

    best_learning_rate = compute_best_logreg_learning_rate(
        train_matrix,
        train_labels,
        val_matrix,
        val_labels,
        LEARNING_RATES,
    )
    best_val_accuracy = compute_accuracy(
        train_matrix, train_labels, val_matrix, val_labels, best_learning_rate)
    test_accuracy = compute_accuracy(
        train_matrix, train_labels, test_matrix, test_labels, best_learning_rate)

    print('Best learning rate: {}'.format(best_learning_rate))
    print('Validation accuracy: {}'.format(best_val_accuracy))
    print('Test accuracy: {}'.format(test_accuracy))


if __name__ == "__main__":
    main()
