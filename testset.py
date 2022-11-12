import numpy as np
import GaussianNaiveBayes as gnb
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

model = gnb.GaussianNaiveBayes()

print(np.mean(train[0:3, 2]))
model.fit(X_train, y_train)





