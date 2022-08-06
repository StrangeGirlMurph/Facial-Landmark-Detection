import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import random


def loadData(includeAugmented=True):
    """ Loads the data and returns the training feautures/labels and test images. """
    print("\n> Loading the data...")

    X_train = np.load('../data/processedData/X_train.npy').astype(np.float32)
    y_train = np.load('../data/processedData/y_train.npy').astype(np.float32)
    X_test = np.load('../data/processedData/X_test.npy').astype(np.float32)

    if includeAugmented:
        X_augmented = np.load('../data/processedData/X_augmented.npy').astype(np.float32)
        y_augmented = np.load('../data/processedData/y_augmented.npy').astype(np.float32)
        return np.concatenate((X_train, X_augmented)), np.concatenate((y_train, y_augmented)), X_test

    return X_train, y_train, X_test


def processRawData(augmentation=True):
    """Reads the data from the bare csv files, preprocesses the data and stores it in numpy files for faster access. (Preprocessing and reading from the csv files all the time is slow)"""
    print("\n> Processing the raw data...")

    # -- read data from csv --
    trainData = pd.read_csv('../../data/rawData/trainingData.csv')
    testData = pd.read_csv('../../data/rawData/testData.csv')

    # -- preprocessing --
    print("- Checking for missing values...")
    print(trainData.isnull().sum())

    cleanTrainData = trainData.dropna(inplace=False)
    # fill missing values with -1 for the masking
    trainData.fillna(value=-1., inplace=True)

    print("- Missing values after filling:")
    print(trainData.isnull().any().value_counts())

    print("- Bringing data into shape...")
    X_train = preprocessFeatures(trainData)
    y_train = preprocessLabels(trainData)
    X_test = preprocessFeatures(testData)
    X_cleanTrain = preprocessFeatures(cleanTrainData)
    y_cleanTrain = preprocessLabels(cleanTrainData)

    print("- Performing augmentations on the clean clean data...")
    X_augmented, y_augmented = performRotationAugmentation(X_cleanTrain, y_cleanTrain, angles=[-12, 12])

    print("Features: ", X_train.shape, X_train.dtype)
    print("Labels: ", y_train.shape, y_train.dtype)
    print("Augmented features: ", X_augmented.shape, X_augmented.dtype)
    print("Augmented labels: ", y_augmented.shape, y_augmented.dtype)

    print("- Storing the data as numpy files...")
    np.save('../../data/processedData/X_train.npy', X_train)
    np.save('../../data/processedData/y_train.npy', y_train)
    np.save('../../data/processedData/X_augmented.npy', X_augmented)
    np.save('../../data/processedData/y_augmented.npy', y_augmented)
    np.save('../../data/processedData/X_test.npy', X_test)

    print("- Done!")


def performRotationAugmentation(X, y, angles=[12]):
    """Performs rotation augmentation on the given data."""
    X_new = np.empty(shape=(0, 96, 96), dtype=np.uint8)
    y_new = np.empty(shape=(0, 30), dtype=np.float16)

    # bring the labels in coordinate shape (x, y, 1)
    c = np.array([np.stack((yi[0::2], yi[1::2]), axis=1) for yi in y])
    c = np.pad(c, ((0, 0), (0, 0), (0, 1)), 'constant', constant_values=(1))

    for angle in angles:
        M = cv.getRotationMatrix2D((48, 48), angle, 1.0)
        X_new = np.concatenate((X_new, [cv.warpAffine(x, M, (96, 96), flags=cv.INTER_CUBIC) for x in X]))
        y_new = np.concatenate((y_new, [np.array([np.matmul(M, yi) for yi in ci], dtype=np.float16).flatten() for ci in c]))

    return X_new.reshape(-1, 96, 96, 1), y_new


def preprocessLabels(data):
    return data.drop('Image', axis=1).to_numpy(dtype=np.float16)


def preprocessFeatures(data):
    images = data['Image'].str.split(" ")
    return np.array(images.to_list(), dtype=np.uint8).reshape(-1, 96, 96, 1)


def generateImages():
    X_train, y_train, X_test = loadData()

    indices = [random.randint(0, len(X_train)-1) for p in range(0, 100)]
    for i in indices:
        fig = plt.figure()
        plt.imshow(X_train[i], cmap="gray")
        plt.scatter(y_train[i][0::2], y_train[i][1::2], c='b', marker='.')
        plt.tight_layout()
        plt.savefig(f"../data/sampleImages/{i}.png", bbox_inches='tight')
        plt.close(fig)
