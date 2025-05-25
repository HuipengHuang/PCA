import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_circles
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

    else:
        raise NotImplemented("Not implemented yet")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=int(y.shape[0] * args.test_ratio), random_state=0)
    return X_train, X_test, y_train, y_test
