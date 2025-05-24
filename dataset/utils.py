import pandas as pd
from sklearn.model_selection import train_test_split

def build_dataset(args):
    if args.dataset == "breast_cancer":
        data = pd.read_csv("./data/breast_cancer/data.csv").dropna(axis=1)
        y = data["diagnosis"].to_numpy()
        mask = y == "M"
        # Label 0 means benign. Label 1 means
        y[mask] = 1
        y[~mask] = 0
        X = data.drop("diagnosis", axis=1).to_numpy()
    else:
        raise NotImplemented("Not implemented yet")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=int(y.shape[0] * args.test_ratio), random_state=0)
    return X_train, X_test, y_train, y_test
