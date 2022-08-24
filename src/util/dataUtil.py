from util.augmentationUtil import rotate, horizontalFlip, cropAndPad, perspective, brightnessAndContrast
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2 as cv
import random


def loadData(includeAugmented=True):
    """ Loads the data and returns the training feautures/labels and test images. """
    print("\n> Loading the data...")

    X_train = np.load('../data/processedData/X_train.npy').astype(np.float32)
    y_train = np.load('../data/processedData/y_train.npy').astype(np.float32)
    X_test = np.load('../data/processedData/X_test.npy').astype(np.float32)

    if includeAugmented:
        X_augmented, y_augmented = loadAugmentedData()
        return np.concatenate((X_train, X_augmented)), np.concatenate((y_train, y_augmented)), X_test

    return X_train, y_train, X_test


def loadAugmentedData():
    """Loads the augmented data from the numpy files."""
    return np.load('../data/processedData/X_augmented.npy').astype(np.float32), np.load('../data/processedData/y_augmented.npy').astype(np.float32)


def performDataAugmentation(X, y, onlyCleanData=True):
    """Performs data augmentation on the given data and stores the data."""
    print("\n> Performing data augmentation...")

    if onlyCleanData:
        a, b = np.where(y == -1)
        indices = np.unique(a)  # indices of the unclean images
        X = np.delete(X, indices, axis=0)
        y = np.delete(y, indices, axis=0)

    y = y.reshape(-1, 15, 2)

    X_augmented, y_augmented = np.empty((0, 96, 96, 1)), np.empty((0, 15, 2))

    augmentations = [rotate, horizontalFlip, cropAndPad, perspective, brightnessAndContrast]
    print(f"- Performing: {', '.join(list(map(lambda a: a.__name__,augmentations)))}")
    for augmentation in augmentations:
        X_new, y_new = np.copy(X), np.copy(y)
        for i in tqdm(np.ndindex(X_new.shape[0])):
            X_new[i], y_new[i] = augmentation(X_new[i], y_new[i])
        X_augmented = np.concatenate((X_augmented, X_new))
        y_augmented = np.concatenate((y_augmented, y_new))

    a, b, c = np.where((y_augmented < 0) | (y_augmented > 96))
    y_augmented[a, b, :] = -1
    y_augmented = y_augmented.reshape(-1, 30).astype(np.float16)
    X_augmented = X_augmented.astype(np.uint8)

    print("Augmented features: ", X_augmented.shape, X_augmented.dtype)
    print("Augmented labels: ", y_augmented.shape, y_augmented.dtype)

    np.save('../data/processedData/X_augmented.npy', X_augmented)
    np.save('../data/processedData/y_augmented.npy', y_augmented)

    print("- Done!")


def processRawData():
    """Reads the data from the bare csv files, preprocesses the data and stores it in numpy files for faster access. (Preprocessing and reading from the csv files all the time is slow)"""
    print("\n> Processing the raw data...")

    # -- read data from csv --
    trainData = pd.read_csv('../data/rawData/trainingData.csv')
    testData = pd.read_csv('../data/rawData/testData.csv')

    # -- preprocessing --
    print("- Checking for missing values...")
    print(trainData.isnull().sum())

    # fill missing values with -1 for the masking
    trainData.fillna(value=-1., inplace=True)

    print("- Missing values after filling:")
    print(trainData.isnull().any().value_counts())

    print("- Bringing data into shape...")
    X_train = preprocessFeatures(trainData)
    y_train = preprocessLabels(trainData)
    X_test = preprocessFeatures(testData)

    print("Features: ", X_train.shape, X_train.dtype)
    print("Labels: ", y_train.shape, y_train.dtype)

    print("- Storing the data as numpy files...")
    np.save('../data/processedData/X_train.npy', X_train)
    np.save('../data/processedData/y_train.npy', y_train)
    np.save('../data/processedData/X_test.npy', X_test)

    print("- Done!")


def preprocessLabels(data):
    """Extracts the labels from the data and returns them in the correct form."""
    return data.drop('Image', axis=1).to_numpy(dtype=np.float16)


def preprocessFeatures(data):
    """Extracts the images from the data and returns them in the correct form."""
    images = data['Image'].str.split(" ")
    return np.array(images.to_list(), dtype=np.uint8).reshape(-1, 96, 96, 1)


def generateImages(X, y, folderName="sampleImages"):
    """Generates png images from the raw data and stores them under "../data/{folderName}/*.png" (the folder in data already has to exist)."""
    indices = [random.randint(0, len(X)-1) for p in range(0, 100)]
    for i in indices:
        fig = plt.figure()
        plt.imshow(X[i], cmap="gray")
        plt.scatter(y[i][0::2], y[i][1::2], c='b', marker='.')
        plt.tight_layout()
        plt.savefig(f"../data/{folderName}/{i}.png", bbox_inches='tight')
        plt.close(fig)


def printDatasetOverview():
    """Prints some basic information about the dataset."""
    print("\n> Printing dataset overview...")
    data = pd.read_csv('../data/rawData/trainingData.csv')

    print("Head:\n", data.head().T[0])
    print("Info:\n", data.info())
