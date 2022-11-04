import numpy as np
import pandas as pd

class GaussianNaiveBayes:
    """
    A class to use Gaussian Naive Bayes to classify data.
    ...
    Parameters
    _________
    classes :

    Attributes
    __________
    n_col : int
        Number of columns in the dataset X

    params_ : array-like, shape = [2, n_col]
        Stores the mean and variance (var) from the data X

    Methods
    _______
    train(X_train, y_train) :
        Creates the parameters mean and var for each category of the data

    test(X_test, y_test) :


    gaussian(x, mean, var):
        Calculates the likelihood for a given attribute given mean, variance and test value

    """

    def fit(self, X, y):
        """
        Finding and storing the parameters from the training data

        Parameters:
            X: array-like, shape = [n_samples, n_features]
                Training vectors
            y: array-like, shape = [n_samples]
                Target values from training set

        Returns:
            self: object
        """

