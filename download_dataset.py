from pathlib import Path
from sklearn.datasets import fetch_openml


def download_dataset():
    """
    This function downloads the MNIST dataset, if not present in the "dataset" folder, otherwise it loads it from the same folder.

    OUTPUT:
    - X, i.e. the feature vector;
    - y, i.e. the label vector.
    """
    X,y = fetch_openml('mnist_784', version=1, return_X_y=True,parser='auto')
    y = y.astype(int)
    X = X/255

    X.to_parquet("dataset/X.parquet")
    y.to_frame().to_parquet("dataset/y.parquet")
    
    return X,y
download_dataset()