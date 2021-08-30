# Import the Logistic Regression Module from Scikit Learn
from sklearn.linear_model import LogisticRegression

# Import the IRIS Dataset to be used in this Kernel
from sklearn.datasets import load_iris

# Load the Module to split the Dataset into Train & Test
from sklearn.model_selection import train_test_split

import joblib

# Load the data
Iris_data = load_iris()

# Split data
Xtrain, Xtest, Ytrain, Ytest = train_test_split(Iris_data.data,
                                                Iris_data.target,
                                                test_size=0.3,
                                                random_state=4)

# Define the Model
LR_Model = LogisticRegression(C=0.1,
                               max_iter=20,
                               fit_intercept=True,
                               n_jobs=3,
                               solver='liblinear')

# Train the Model
LR_Model.fit(Xtrain, Ytrain)

# Save RL_Model to file in the current working directory

joblib_file = "joblib_RL_Model.pkl"
joblib.dump(LR_Model, joblib_file)

joblib_LR_model = joblib.load(joblib_file)