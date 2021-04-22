# libraries
import numpy as np
from collections import Counter

#
class KNNClassifier():
    """
    K-Nearest-Neighbor Machine Learning Algorithm for Classification,
    which computes the k nearest neighbors for one or more chosen data points using the Euclidean distance
    and determines the most occured class label according to these neighbors via majority voting.
    The most occuring class is the predicted class for a chosen data point. 
    """
    def __init__(self, k: int=3):
        """
        Constructor to initialize the number of the nearest neighbors.
        """
        self.k = k

    @staticmethod
    def euclidean_distance(p: np.ndarray, q: np.ndarray):
        """
        Static method for the computation of the Euclidean distance between two data points.
        """
        return np.linalg.norm(p - q)

    def _k_nearest_neighbors(self, X_new: np.ndarray):
        """
        Protected method for the computation of the k-nearest neighbors of one or more sample x-data.
        """
        knn_idx_array = np.array([np.argsort([KNNClassifier.euclidean_distance(p_i, q_i) for q_i in self.X])[:self.k] for p_i in X_new])
        return knn_idx_array

    def _class_majority_voting(self, knn_idx: np.ndarray):
        """
        Protected method for choosing the class for one or more data points by taking this class which occured most
        during the k-nearest neighbor computation (= majority voting).
        """
        class_labels_list = [self.y[i] for i in knn_idx]
        major_class = Counter(class_labels_list).most_common(1)
        return major_class[0][0]

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Initializing X-data and y-data.
        """
        self.X = X
        self.y = y

    def predict(self, X_new: np.ndarray):
        """
        Predicts the class for one or more new data points.
        """
        knn_idx_array = self._k_nearest_neighbors(X_new)
        class_predictions = np.array([self._class_majority_voting(i) for i in knn_idx_array])
        return class_predictions

    def accuracy(self, y_true: np.ndarray, y_predictions: np.ndarray):
        """
        Computation of the accuracy of the model.
        """
        acc = np.sum([y_true_i == y_pred_i for y_true_i, y_pred_i in zip(y_true, y_predictions)]) / len(y_true)
        return acc

