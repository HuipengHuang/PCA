import numpy as np
from .base_pca import BasePCA
from sklearn.metrics.pairwise import pairwise_kernels


class KernelPCA(BasePCA):
    def __init__(self, args):
        """
        Initialize Kernel PCA.

        Args:
            n_components (int): Number of principal components to keep
            kernel (str): Kernel type ('rbf', 'poly', 'linear')
            gamma (float): Kernel coefficient for 'rbf' and 'poly' kernels
        """
        super().__init__(args)
        self.kernel = args.kernel
        self.gamma = args.gamma
        self.X_fit = None
        self.alphas = None
        self.eg_vectors = None

        self.lambdas = None
        self.eg_values = None

        self.mean_kernel = None

    def _kernel(self, X, Y=None):
        """Compute kernel matrix"""
        return pairwise_kernels(X.T, Y=Y if Y is not None else X, metric=self.kernel, gamma=self.gamma)

    def fit(self, X):
        """
        Fit the Kernel PCA model to the data.

        Args:
            X (np.ndarray): Input data, shape (n_samples, n_features)
        """
        self.X_fit = X

        # Compute kernel matrix
        kernel_X = self._kernel(X)

        # Center the kernel matrix
        K_centered = self.center(kernel_X)
        self.mean_kernel = K_centered

        # Compute eigenvalues and eigenvectors
        eg_value, eg_vectors = np.linalg.eigh(K_centered)

        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eg_value)[::-1]
        eg_value = eg_value[idx]
        eg_vectors = eg_vectors[:, idx]

        # Store first n_components components
        if self.n_components is None:
            self.n_components = X.shape[1]

        self.eg_values = eg_value
        self.eg_vectors = eg_vectors

        # Normalize eigenvectors
        print("haha")
        self.eg_vectors = self.eg_vectors / np.sqrt(np.sum(self.eg_vectors * self.eg_vectors, axis=0))
        print(self.eg_vectors.shape)
        print(self.eg_vectors @ self.eg_vectors.T)
    def transform(self, X, n_components=None):
        """
        Project data onto the kernel principal components.

        Args:
            X (np.ndarray): Data to transform, shape (n_samples, n_features)
            n_components (int, optional): Number of components to use

        Returns:
            np.ndarray: Transformed data, shape (n_samples, n_components)
        """
        if n_components is None:
            n_components = self.n_components

        # Compute kernel matrix between new data and training data
        K_test = self._kernel(X, self.X_fit)

        K_test_centered = self.center(K_test)

        # Project onto principal components
        X_transformed = np.dot(K_test_centered, self.eg_vectors[:, :n_components])

        return X_transformed

    def center(self, the_matrix):
        n_rows = the_matrix.shape[0]
        n_cols = the_matrix.shape[1]
        vector_one_left = np.ones((n_rows,1))
        vector_one_right = np.ones((n_cols, 1))
        H_left = np.eye(n_rows) - ((1/n_rows) * vector_one_left.dot(vector_one_left.T))
        H_right = np.eye(n_cols) - ((1 / n_cols) * vector_one_right.dot(vector_one_right.T))

        mode = self.args.mode
        if mode == "double_center":
            the_matrix = H_left.dot(the_matrix).dot(H_right)
        elif mode == "remove_mean_of_rows_from_rows":
            the_matrix = H_left.dot(the_matrix)
        elif mode == "remove_mean_of_columns_from_columns":
            the_matrix = the_matrix.dot(H_right)

        return the_matrix