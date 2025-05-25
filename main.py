import argparse
import matplotlib.pyplot as plt
from dataset.utils import build_dataset
from PCA.utils import get_pca, visualize_results, save_grayscale_array_as_video


parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=None, type=int)
parser.add_argument("--save", default="False", type=str)

parser.add_argument("--pca", default="robust_pca", choices=["pca", "ppca", "kernel_pca", "sparse_pca", "ppca", "robust_pca"])
parser.add_argument("--dataset", default="video", type=str, choices=["yaleB", "make_circle", "make_moon", "breast_cancer", "ext_yaleB", "video"])
parser.add_argument("--test_ratio", default=None, type=float)
parser.add_argument("--n_components", type=int, default=10)

#hyperparameter for kernel pca
parser.add_argument("--kernel", type=str, default="rbf",
                    choices=['additive_chi2', 'chi2', 'linear', 'poly', 'polynomial', 'rbf', 'laplacian', 'sigmoid', 'cosine'])
parser.add_argument("--gamma", type=float, default=10)

#hyperparameter for sparse_pca
parser.add_argument("--max_iter", type=int, default=1000)
parser.add_argument("--tol", type=float, default=None)

args = parser.parse_args()

X_train, X_test, y_train, y_test = build_dataset(args)

pca = get_pca(args)
pca.fit(X_train)

if args.pca != "robust_pca":
    transformed_X = pca.transform(X_train)
    visualize_results(X_train, transformed_X, y_train, args)
else:
    L, S = pca.transform(X_train)
    if args.dataset == "yaleB":
        plt.imshow(X_train[0].reshape(243, 320), cmap='gray')
        plt.axis('off')
        plt.show()

        plt.imshow(L[0].reshape(243, 320), cmap='gray')
        plt.axis('off')
        plt.show()

        plt.imshow(S[0].reshape(243, 320), cmap='gray')
        plt.axis('off')
        plt.show()
    elif args.dataset == "ext_yaleB":
        fig, ax = plt.subplots()
        ax.imshow(X_train[1].reshape(168, 192), cmap='gray')
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove padding
        plt.savefig(f"./output/ext_yaleB/{args.pca}_origin.png")
        plt.show()

        fig, ax = plt.subplots()
        ax.imshow(L[1].reshape(168, 192), cmap='gray')
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(f"./output/ext_yaleB/{args.pca}_L.png")
        plt.show()

        fig, ax = plt.subplots()
        ax.imshow(S[1].reshape(168, 192), cmap='gray')
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(f"./output/ext_yaleB/{args.pca}_S.png")
        plt.show()
    elif args.dataset == "video":
        print("Generating video")
        X_train = X_train.reshape(-1, 282, 378)
        L = L.reshape(-1, 282, 378)
        S = S.reshape(-1, 282, 378)
        save_grayscale_array_as_video(X_train, output_path="./output/video/origin.mp4")
        save_grayscale_array_as_video(L, output_path="./output/video/L.mp4")
        save_grayscale_array_as_video(S, output_path="./output/video/S.mp4")
#visualize_results(X_train, transformed_X, y_train, args)
