import numpy as np


def loadData():
    """ Loads the data and returns the training feautures/labels and test images. """
    print("\n> Loading the data...")

    X_train = np.load('../data/processedData/X_train.npy').astype(np.float32)
    y_train = np.load('../data/processedData/y_train.npy')
    X_test = np.load('../data/processedData/X_test.npy').astype(np.float32)

    return X_train, y_train, X_test
