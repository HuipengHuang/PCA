import os.path
import numpy as np
from .pca import PCA
from .kernel_pca import KernelPCA
from .sparese_pca import SparsePCA
from .ppca import ProbabilisticPCA
from .robust_pca import RobustPCA
import matplotlib.pyplot as plt




def get_pca(args):
    if args.pca == "pca":
        return PCA(args)
    elif args.pca == "kernel_pca":
        return KernelPCA(args)
    elif args.pca == "sparse_pca":
        return SparsePCA(args)
    elif args.pca == "ppca":
        return ProbabilisticPCA(args)
    elif args.pca == "robust_pca":
        return RobustPCA(args)
    else:
        raise NotImplementedError

def visualize_results(X_original, X_transformed, y, args):
    """Plot original vs transformed data with customized styling"""
    # Set common style parameters
    title_font = {'fontsize': 25, 'fontweight': 'bold', 'color': 'black', "fontname": "Times New Roman"}
    frame_width = 3.0  # Bold frame width

    # Plot 1: Original Data
    plt.figure(figsize=(8, 5))

    # Create scatter plot with frame
    ax = plt.gca()
    ax.scatter(X_original[:, 0], X_original[:, 1], c=y, cmap='viridis', alpha=0.6)

    # Customize frame
    for spine in ax.spines.values():
        spine.set_linewidth(frame_width)

    # Add title at bottom
    ax.set_title("Original Data", fontdict=title_font, y=-0.12, pad=-10)

    plt.tight_layout()

    if args.save == "True":
        os.makedirs("./output", exist_ok=True)
        save_path = f"./output/{args.dataset}/original_data_{args.n_components}.pdf"

        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()

    # Plot 2: Transformed Data
    plt.figure(figsize=(8, 4))
    ax = plt.gca()

    if X_transformed.shape[1] == 1:
        ax.scatter(X_transformed[:, 0], np.zeros_like(X_transformed[:, 0]),
                   c=y, cmap='viridis', alpha=0.6,
                   s=200)  # Larger points (no frame)
    else:
        ax.scatter(X_transformed[:, 0], X_transformed[:, 1],
                   c=y, cmap='viridis', alpha=0.6,
                   s=400)  # Larger points (no frame)

    # Customize frame
    for spine in ax.spines.values():
        spine.set_linewidth(frame_width)

    # Add title at bottom
    if args.pca == "pca":
        ax.set_title(f"Naive PCA (n={args.n_components})",
                     fontdict=title_font, y=-0.12, pad=-20)
    elif args.pca == "kernel_pca":
        ax.set_title(f"Kernel PCA (kernel={args.kernel}, γ={args.gamma}, n={args.n_components})",
                     fontdict=title_font, y=-0.12, pad=-50)
    else:
        raise NotImplementedError

    plt.tight_layout()

    if args.save == "True":
        i = 0
        save_path = f"./output/{args.dataset}/{args.kernel}_{args.gamma}_{args.n_components}_{i}.pdf"
        while os.path.exists(save_path):
            i += 1
            save_path = f"./output/{args.dataset}/{args.kernel}_{args.gamma}_{args.n_components}_{i}.pdf"
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()