import numpy as np
import statistics as stat
import math

class GaussianNaiveBayes:
    """
    A class to use Gaussian Naive Bayes to classify data.
    ...
    Parameters
    _________
    prior : string
        The prior distribution of the classes.

    Attributes
    __________
    n_cols_: int
        Number of columns in the training dataset X
        Represents the number of unique attributes in our data set

    n_labels_: int
        Number of unique class labels in the training set y
        Represents the number of categories we are sorting our data into

    priors_: array-like, shape = [1, n_labels]


    data_: list of array-likes, shape = [n_labels, varies, n_cols]
        A list of matrices that contain the training data sorted by class label

    params_ : array-like, shape = [1, 2, n_cols]
        Stores the mean and variance (var) from the data X

    Methods
    _______
    train(X_train, y_train) :
        Creates the parameters mean and var for each category of the data

    test(X_test, y_test) :
        Uses the parameters from train to predict labels and checks for accuracy

    gaussian(x, mean, var):
        Calculates the likelihood for a given attribute given mean, variance and test value

    """

    def __init__(self, prior='uniform'):
        self.prior = prior

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
        #Sorts the data into a list of arrays data.
        #Each array in data is the training set for a given class label
        self.sort_data(self, X, y)

        # Calculate mean and variance of data and store in params
        for i in range(0, len(self.data_)):
            for j in range(0, self.data_[i].shape[0]):
                self.params_[i, 0, j] = np.mean(self.data_[i][:, j])
                self.params_[i, 1, j] = np.var(self.data_[i][:, j])
        return self

    def test(self, X, y):
        # Creating likelihoods for each class label
        likelihoods = []
        for i in range(0, self.n_labels):
            temp_likelihood = np.zeros(1, X.shape[1])
            for j in range(0, self.data_[i].shape[0]):
                for k in range(0, self.data_[i].shape[1]):
                    temp_likelihood[j] = self.gaussian(X[j, k], self.params_[i, 0, j], self.params_[i, 1, j])



    def sort_data(self, X, y):
        # We first need to separate our data by class label
        labels = np.unique(y)
        self.n_labels = len(labels)
        n_cols = X.shape[1]

        # Creates n_labels matrices holding the mean and variance for each column
        self.params_ = np.zeros((self.n_labels, 2, n_cols))

        # Data will contain arrays of training vectors separated by class
        # Data is a list of numpy arrays
        self.data_ = []
        for i in range(0, len(labels)):
            X_temp = []
            for j in range(0, len(y)):
                if y[j] == labels[i]:
                    X_temp = np.vstack(X_temp, X[i, :])
            self.data_.append(X_temp)
        return self

    def gaussian(self, x, mean, var):
        return (1/math.sqrt(2*math.pi*var))*math.exp(-(x-mean)*(x-mean)/(2*var))

    def create_prior(self):
        if self.prior=='uniform':
            return

