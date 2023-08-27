
# Import libraries
import pandas as pd
import numpy as np
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import pickle
from pickle import dump
from pickle import load

# Assign column names
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

# Load the data and rename columns
df = read_csv(url, names=names)

# Assign target and independent variables
myarray = df.values
Y = myarray[:,8]
X = myarray[:,0:8]

# Set kfold parameter
kfold = KFold(n_splits=10, shuffle=True, random_state=7)

# Create test and train split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=7)

# Fit the model on train dataset using LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# Score
# scoring = 'accuracy'
# results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
# print("Accuracy Mean and Standard Deviation:", results.mean(), results.std())

# Test Predict and confusion_matrix
# predicted = model.predict(X_test)
# matrix = confusion_matrix(Y_test, predicted)
# print(matrix)


pickle.dump(model, open('pima.pkl', 'wb'))