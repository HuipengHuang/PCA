import numpy as np

from .base_pca import BasePCA
class PCA(BasePCA):
    def __init__(self, args):
        super().__init__(args)
        self.U = None
        self.eigen_value = None
        self.V = None

    def fit(self, X):
        """
        Fit the PCA model to the data using SVD.

        Args:
            X (np.ndarray): Input data, shape (n_samples, n_features).
        """

        self.mean = np.mean(X, axis=0)

        X_centered = X - self.mean[np.newaxis, :]

        # Perform SVD on centered data
        U, singular_values, V_T = np.linalg.svd(X_centered, full_matrices=False)
        self.U = U
        print(singular_values)
        self.eigen_value = singular_values ** 2
        self.V = V_T.T

        if self.n_components is None:
            self.n_components = min(X.shape[0], X.shape[1])

    def transform(self, X, n_components=None):
        """
        Project data onto the principal components.

        Args:
            X (np.ndarray): Data to transform, shape (n_samples, n_features).
            n_components (int, optional): Number of components to use. Defaults to self.n_components.

        Returns:
            np.ndarray: Transformed data, shape (n_samples, n_components).
        """
        n = n_components if n_components is not None else self.n_components

        # Center the input data across features for each sample
        X_centered = X - self.mean[np.newaxis, :]

        # Project data onto the principal components
        X_transformed = np.dot(X_centered, self.V[:, :n])

        return X_transformed
