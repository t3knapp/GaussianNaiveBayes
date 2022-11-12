import numpy as np
import GaussianNaiveBayes as GNB
import pandas as pd

# Training Set importing as numpy array
train = pd.read_csv('Gender/Train.csv').to_numpy()
# Separating Class labels and data
X_train = train[:, 1:]
y_train = train[:, 0]

# Test Set importing as numpy array
test = pd.read_csv('Gender/Test.csv').to_numpy()
# Separating Class labels and data
X_test = test[:, 1:]
y_test = test[:, 0]

# Creating GNB Object
model = GNB.GaussianNaiveBayes()

# Fitting the object to the training data
model.fit(X_train, y_train)

# Testing the model
out = model.predict(X_test, y_test)

print('Class predictions given by gaussian naive bayes:')
print(out)






