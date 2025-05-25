import argparse

import numpy as np

from dataset.utils import build_dataset
from PCA.utils import get_pca, visualize_results
parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=None, type=int)
parser.add_argument("--save", default="False", type=str)

parser.add_argument("--pca", default="robust_pca", choices=["pca", "ppca", "kernel_pca", "sparse_pca", "ppca", "robust_pca"])
parser.add_argument("--dataset", default="breast_cancer", type=str)
parser.add_argument("--test_ratio", default=0.2, type=float)
parser.add_argument("--n_components", type=int, default=10)

#hyperparameter for kernel pca
parser.add_argument("--kernel", type=str, default="rbf",
                    choices=['additive_chi2', 'chi2', 'linear', 'poly', 'polynomial', 'rbf', 'laplacian', 'sigmoid', 'cosine'])
parser.add_argument("--gamma", type=float, default=10)

#hyperparameter for sparse_pca
parser.add_argument("--max_iter", type=int, default=100)
parser.add_argument("--tol", type=float, default=None)

args = parser.parse_args()

X_train, X_test, y_train, y_test = build_dataset(args)

pca = get_pca(args)
pca.fit(X_train)

if args.pca != "robust_pca":
    transformed_X = pca.transform(X_train)
    print(X_train.shape)
    print(transformed_X.shape)
else:
    L, S = pca.transform(X_train)
    print(X_train.shape)
    print(L.shape)
    print(S.shape)


#visualize_results(X_train, transformed_X, y_train, args)
