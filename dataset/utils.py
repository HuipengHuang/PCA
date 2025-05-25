import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_circles
from skimage.io import imread, imsave
import os

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
        X = np.array(image_list)[:2]
        y = None
    elif args.dataset == "video":
        dir_path = "./data/video/JPEGS/traffic"
        image_list = []

        for filename in os.listdir(dir_path):
            image_path = os.path.join(dir_path, filename)
            image = imread(image_path)
            image_list.append(image.reshape(-1))
        X = np.array(image_list)
        y = None
    else:
        raise NotImplemented("Not implemented yet")

    if args.test_ratio:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=int(y.shape[0] * args.test_ratio), random_state=0)
        return X_train, X_test, y_train, y_test
    else:
        return X, None, y, None
