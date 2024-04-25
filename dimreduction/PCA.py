import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class PCA:
    """
    PCA class to perform PCA on a dataset.
    """
    def __init__(self, n_components):
        """
        Initialize PCA instance.
        :param n_components: Number of principal components to keep.
        """
        self.n_components = n_components
        self.standardized = None
        self.eigvecs = None

    @staticmethod
    def standardize(X):
        """
        Standardize the dataset.
        :param X: dataset to standardize.
        :return: standardized dataset.
        """
        X_mean = X.mean()
        X_std = X.std()
        zero_std_cols = X_std[X_std == 0].index
        X_stand = (X - X_mean) / X_std
        X_stand.loc[:, zero_std_cols] = 0
        return X_stand

    def fit(self, X):
        """
        Calculate PCA to the dataset.
        :param X: dataset to perform PCA on.
        :return: None
        """
        X_stand = PCA.standardize(X)
        self.standardized = X_stand
        X_cov = X_stand.cov()
        eigvals, eigvecs = np.linalg.eigh(X_cov)
        eigvecs = eigvecs[:, ::-1]
        self.eigvecs = eigvecs[:, :self.n_components]

    def transform(self, X):
        """
        Transform the dataset to the new space.
        :param X: dataset to transform, should be standardized.
        :return: transformed dataset.
        """
        return np.dot(X, self.eigvecs)

    def visualize(self, X, Y):
        """
        Visualize the transformed dataset.
        :param X: dataset to visualize, should be standardized.
        :param Y: target labels.
        :return: None
        """
        X_transformed = self.transform(X)
        plt.figure(figsize=(30, 30))
        for i in np.unique(Y):
            plt.scatter(X_transformed[Y == i, 0], X_transformed[Y == i, 1], label=i)
        plt.legend()
        plt.show()
