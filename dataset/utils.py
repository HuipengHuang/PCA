import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_circles
from skimage.io import imread, imsave
import os
import cv2

def build_dataset(args):
    if args.dataset == "breast_cancer":
        data = pd.read_csv("./data/breast_cancer/data.csv").dropna(axis=1)
        y = data["diagnosis"].to_numpy()
        mask = y == "M"
        # Label 0 means benign. Label 1 means M
        y[mask] = 1
        y[~mask] = 0
        X = data.drop("diagnosis", axis=1).to_numpy()
    elif args.dataset == "make_moon":
        X, y = make_moons(shuffle=True, random_state=args.seed if args.seed else None, n_samples=5000, noise=0.1)
    elif args.dataset == "make_circle":
        X, y = make_circles(shuffle=True, random_state=args.seed if args.seed else None, n_samples=5000, factor=0.2, noise=0.1)
    elif args.dataset == "yaleB":
        rand = random.randint(0, 14)
        rand = str(rand)
        rand = str(3)
        if len(rand) == 1:
            rand = "0" + rand
        image_list = []
        dir_path = "./data/yaleB"
        for filename in os.listdir(dir_path):
            if f"subject{rand}" in filename:
                image_path = os.path.join(dir_path, filename)
                image = imread(image_path)
                image_list.append(image.reshape(-1))  # Add to list
        X = np.array(image_list)
        y = None
    elif args.dataset == "ext_yaleB":
        dir_path = "./data/ext_yaleB/cropped/yaleB12"
        image_list = []
        for filename in os.listdir(dir_path):
            image_path = os.path.join(dir_path, filename)
            image = imread(image_path)
            image_list.append(image.reshape(-1))
        X = np.array(image_list)
        y = None
    elif args.dataset == "video":
        dir_path = "./data/video/JPEGS/traffic"
        image_list = []
        i = 1
        filename = f"frame_{i}.jpg"
        while os.path.isfile(os.path.join(dir_path, filename)):
            image_path = os.path.join(dir_path, filename)
            image = imread(image_path)
            image_list.append(image.reshape(-1))
            i += 1
            filename = f"frame_{i}.jpg"
        X = np.array(image_list)
        print(X.shape)
        y = None
    elif args.dataset == "sustech_video":
        dir_path = "./data/sustech_video/sustech.mp4"
        video_array = video_to_array(dir_path)
        X = video_array.reshape(video_array.shape[0], -1)
        y = None
        print(X.shape)
    else:
        raise NotImplemented("Not implemented yet")

    if args.test_ratio:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=int(y.shape[0] * args.test_ratio), random_state=0)
        return X_train, X_test, y_train, y_test
    else:
        return X, None, y, None


def video_to_array(video_path, max_frames=None, target_size=None):
    """
    Convert video to numpy array.

    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to load (None for all)
        target_size: Optional (width, height) to resize frames

    Returns:
        numpy array of shape (num_frames, height, width, 3) for RGB
        or (num_frames, height, width) for grayscale
    """
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video {video_path}")

    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Adjust for max_frames
    if max_frames is not None:
        frame_count = min(frame_count, max_frames)

    # Initialize array
    if target_size:
        h, w = target_size[1], target_size[0]
    else:
        h, w = height, width

    video_array = np.empty((frame_count, h, w, 3), dtype=np.uint8)

    # Read frames
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize if needed
        if target_size:
            frame_rgb = cv2.resize(frame_rgb, target_size)

        video_array[i] = frame_rgb

    cap.release()

    # Remove unused preallocated space if video was shorter than expected
    if i + 1 < frame_count:
        video_array = video_array[:i + 1]

    return video_array
