from .pca import PCA
from .kernel_pca import KernelPCA
def get_pca(args):
    if args.pca == "pca":
        return PCA(args)
    elif args.pca == "kernel_pca":
        return KernelPCA(args)