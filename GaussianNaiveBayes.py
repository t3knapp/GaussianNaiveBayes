import numpy as np

class GaussianNaiveBayes:
    """
    A class to use Gaussian Naive Bayes to classify data.
    ...
    Parameters
    _________
    prior_ : string
        The prior distribution of the classes.

    Attributes
    __________
    n_cols_: int
        Number of columns in the training dataset X
        Represents the number of unique attributes in our data set

    n_labels_: int
        Number of unique class labels in the training set y
        Represents the number of categories we are sorting our data into

    labels_: array like, shape = [1, n_labels]
        Unique labels from the training set

    prior_type_: array-like, shape = [1, n_labels]
        Prior distribution of our test data
        We assume this is uniform for now

    data_: list of array-likes, shape = [n_labels, varies, n_cols]
        A list of matrices that contain the training data sorted by class label

    params_ : array-like, shape = [1, 2, n_cols]
        Stores the mean and variance (var) from the data X

    Methods
    _______
    train(X_train, y_train) :
        Creates the parameters mean and var for each category of the data

    predict(X_test, y_test) :
        Uses the parameters from train to predict labels and checks for accuracy

    gaussian(x, mean, var):
        Calculates the likelihood for a given attribute given mean, variance and test value

    """

    def __init__(self, prior_type_='uniform'):
        self.prior_type_ = prior_type_

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
        # Sorts the data into a list of arrays data.
        # Each array in data is the training set for a given class label
        data_ = self._sort_data(X, y)

        # Calculate mean and variance of data and store in params
        for i in range(0, len(data_)):
            for j in range(0, data_[i].shape[1]):
                self.params_[i, 0, j] = np.mean(data_[i][:, j])
                self.params_[i, 1, j] = np.var(data_[i][:, j])
        return self

    def predict(self, X, y):
        """
                Testing our stored parameters using a test dataset

                Parameters:
                    X: array-like, shape = [n_samples, n_features]
                        Test vectors
                    y: array-like, shape = [n_samples]
                        Target values from test set

                Returns:
                    predict: array-like, shape = [n_samples]
                        Prediction associated with each test vector
                """
        # If prior is uniform, create the prior

        # Creating likelihoods for each class label
        # likelihoods shape = [n_labels][n_samples, n_features]
        likelihoods = []
        for i in range(0, self.n_labels_):
            mean = self.params_[i, 0, :]
            var = self.params_[i, 1, :]
            gaussian = (1/np.sqrt(2*np.pi*var))*np.exp(-(X-mean)*(X-mean)/(2*var))
            likelihoods.append(gaussian)

        # Using likelihoods and prior with N-bayes formula
        # Post is a list of the posterior probabilities separated by label
        post = np.zeros(X.shape[0], self.n_labels_)
        predict = np.zeros(X.shape[0], 1)
        for i in range(0, self.n_labels_):
            # Multiply across rows of likelihoods and multiply by the prior
            post[:, i] = self.prior_[i] * np.prod(likelihoods[i], axis=0)
            # Find the largest posterior and use index to tie it to a label
            predict[i] = self.labels_[np.argmax(post[:, i])]

        # Calculating the percentage of correct predictions
        correct = 0
        for i in range(0, len(y)):
            if predict[i] == y[i]:
                correct += 1

        percent_correct = correct / len(y)
        print("The prediction is " + str(percent_correct) + "% correct")
        return predict

    def _sort_data(self, X, y):
        """
                Takes the training data and separates it by label

                Parameters:
                    X: array-like, shape = [n_samples, n_features]
                        Training vectors
                    y: array-like, shape = [n_samples]
                        Target values from training set

                Returns:
                    data_: list of array-likes, shape = [n_labels][n_samples_in_label, n_features]
                        A list of np arrays, containing the training data separated by label in corresponding y
                """
        # We first need to separate our data by class label
        self.labels_ = np.unique(y)
        self.n_labels_ = len(self.labels_)
        n_cols = X.shape[1]

        # Creates n_labels matrices holding the mean and variance for each column
        self.params_ = np.zeros((self.n_labels_, 2, n_cols))

        X = np.append(X, y, axis=1)
        # Data will contain arrays of training vectors separated by class
        # Data is a list of numpy arrays
        data_ = []
        for i in range(0, len(self.labels_)):
            mask = (X[:, -1] == self.labels_[i])
            data_.append(X[mask, :])
        return data_

    def _create_prior(self):
        """
                Creates the prior distribution from string user input
                Currently only works for a uniform distribution

                Parameters:
                    self: object

                Returns:
                    self: object
                """
        if self.prior_type_ == 'uniform':
            self.prior_ = (1 / self.n_labels_) * np.ones((1, self.n_labels_))
        else:
            raise Exception("Must set prior distribution to uniform.")
        return self
