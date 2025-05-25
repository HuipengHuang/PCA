import os
import cv2
import numpy as np
from tqdm import tqdm
from .base_pca import BasePCA


class RobustPCA(BasePCA):
    def __init__(self, args):
        """
        Initialize Robust PCA (Principal Component Pursuit via ADM).
        Args:
            lambda_ (float): Weight on sparse term (default: 1/sqrt(max_dim)).
            mu (float): Augmented Lagrangian parameter (default: 1e-5).
            rho (float): Multiplicative update for mu (default: 1.1).
            max_iter (int): Maximum iterations (default: 1000).
            tol (float): Convergence tolerance (default: 1e-7).
        """
        super().__init__(args)
        self.lambda_ = None
        self.mu = 0.1
        self.rho = 1.1
        self.max_iter = getattr(args, 'max_iter', 1000)
        self.tol = args.tol if args.tol else 1e-6
        self.L = None  # Low-rank component
        self.S = None  # Sparse component
        self.Y = None

    def fit(self, M):
        """
        Solve RPCA via ADM: min ||L||_* + λ||S||_1 s.t. M = L + S.
        """
        print(M.dtype)
        m, n = M.shape
        self.lambda_ = 1.0 / np.sqrt(max(m, n))  # Default from Candès et al.
        self.mu = 1.25 / np.linalg.norm(M, 2)

        # Initialize variables
        self.L = M.copy()
        self.S = np.zeros_like(M)
        self.Y = np.zeros_like(M)

        k = 0
        for i in tqdm(range(self.max_iter), desc=f"{k+1} / {self.max_iter}"):
            if((i + 1) % 100 == 0 ):
                self.save_video()
            k += 1
            # Step 1: Update L
            res = M - self.S - 1 / self.mu * self.Y
            self.L = self.sv_threshold(res)

            # Step 2: Update S
            residual = M - self.L + self.Y / self.mu
            self.S = self.shrinkage_operator(residual, self.lambda_ * self.mu)

            # Step 3: Update Y
            self.Y = self.Y + self.mu * (M - self.L - self.S)
            self.mu = self.mu * self.rho

            err = self.frobenius_norm(M - self.L - self.S)
            if err < self.tol:
                break

    def transform(self, X, n_components=None):
        """Project data onto the low-rank subspace."""
        L_uint8 = np.clip(self.L, 0, 255).astype(np.uint8)
        S_uint8 = np.clip(self.S, 0, 255).astype(np.uint8)
        return L_uint8, S_uint8
        #return self.L, self.S


    def shrinkage_operator(self, x, threshold):
        sgn = np.sign(x)

        res = (np.abs(x) - threshold)
        relu_output = (res > 0).astype(int) * (res)

        return sgn * relu_output

    def sv_threshold(self, X):
        U, eg_value, V_T = np.linalg.svd(X, full_matrices=False)
        new_eg_value = self.shrinkage_operator(eg_value, self.mu)
        return U @ np.diag(new_eg_value) @ V_T

    def frobenius_norm(self, M):
        return np.linalg.norm(M, ord='fro')

    def save_video(self):
        L = self.L.reshape(-1, 282, 378)
        S = self.S.reshape(-1, 282, 378)
        L = np.clip(L, 0, 255).astype(np.uint8)
        S = np.clip(S, 0, 255).astype(np.uint8)

        i = 0
        output_path = f"./output/video/origin_{i}.mp4"
        while os.path.exists(output_path):
            i += 1
            output_path = f"./output/video/origin_{i}.mp4"


        output_path = f"./output/video/L_{i}.mp4"
        while os.path.exists(output_path):
            i += 1
            output_path = f"./output/video/L_{i}.mp4"

        save_grayscale_array_as_video(L, output_path=output_path)

        output_path = f"./output/video/S_{i}.mp4"
        while os.path.exists(output_path):
            i += 1
            output_path = f"./output/video/S_{i}.mp4"
        save_grayscale_array_as_video(S, output_path=output_path)

def save_grayscale_array_as_video(video_array, output_path, fps=30):
    height, width = video_array.shape[1:3]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in video_array:
        if frame.dtype != np.uint8:

            frame = frame.astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame_bgr)

    out.release()