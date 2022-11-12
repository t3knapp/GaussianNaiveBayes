import GaussianNaiveBayes as GNB
import pandas as pd
import numpy as np

df = pd.read_csv('Stroke/healthcare-dataset-stroke-data.csv')

# We need to clean the data
# Getting rid of the ID column
df.drop('id', inplace=True, axis=1)

# Transforming gender entries into integer values
df.gender.replace(('Female', 'Male'), (0, 1), inplace=True)

# Transforming married status to integer values
df.ever_married.replace(('No', 'Yes'), (0, 1), inplace=True)

# Transforming work type to integer values
df.work_type.replace(df.work_type.unique(), range(0, len(df.work_type.unique())), inplace=True)

# Transforming residence type to integer values
df.Residence_type.replace(df.Residence_type.unique(), range(0, len(df.Residence_type.unique())), inplace=True)

# Transforming Smoking status to integer values
df.smoking_status.replace(df.smoking_status.unique(), range(0, len(df.smoking_status.unique())), inplace=True)

# Removing all rows with empty or NaN entries
df = df.apply(pd.to_numeric, errors='coerce')
df.dropna()


# df should now be all numerical, and okay to use GNB on
# First let's split df into a test and train set.
msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]

print('The training data set will contain: ' + str(len(train)) + ' datapoints.')
print('The testing data set will contain: ' + str(len(test)) + ' datapoints.')

# Convert to numpy array and split into data and labels
train = train.to_numpy()
X_train = train[:, 0:(train.shape[1] - 1)]
y_train = train[:, -1]

test = test.to_numpy()
X_test = test[:, 0:(train.shape[1] - 1)]
y_test = test[:, -1]

model = GNB.GaussianNaiveBayes()

# Fitting the object to the training data
model.fit(X_train, y_train)

# Testing the model
out = model.predict(X_test, y_test)
