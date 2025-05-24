import argparse
from dataset.utils import build_dataset
from PCA.utils import get_pca
parser = argparse.ArgumentParser()
parser.add_argument("--pca", default="kernel_pca", choices=["pca", "ppca", "kernel_pca"])
parser.add_argument("--dataset", default="breast_cancer", type=str)
parser.add_argument("--test_ratio", default=0.2, type=float)
parser.add_argument("--n_components", type=int, default=10)

#hyperparameter for kernel pca
parser.add_argument("--kernel", type=str, default="rbf",
                    choices=['additive_chi2', 'chi2', 'linear', 'poly', 'polynomial', 'rbf', 'laplacian', 'sigmoid', 'cosine'])
parser.add_argument("--gamma", type=float, default=0.1)
parser.add_argument("--mode", type=str, default="remove_mean_of_rows_from_rows",
                    choices=["remove_mean_of_rows_from_rows", "double_center", "remove_mean_of_columns_from_columns"])
args = parser.parse_args()

X_train, X_test, y_train, y_test = build_dataset(args)

pca = get_pca(args)
pca.fit(X_train)
transformed_X = pca.transform(X_train)
print(X_train.shape)
print(transformed_X.shape)
