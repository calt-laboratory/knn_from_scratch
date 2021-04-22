# import libraries
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from k_nearest_neighbor_classifier import *

# build Iris data set
iris_dataset = datasets.load_iris()
X, y = iris_dataset.data, iris_dataset.target

# split dataset into train set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=37)

# use knn classifier
clf = KNNClassifier(k=5)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
print(f"True classes: {y_test}")
print(f"Predicted classes: {preds}")
print(f"Accuracy: {clf.accuracy(y_test, preds)}")