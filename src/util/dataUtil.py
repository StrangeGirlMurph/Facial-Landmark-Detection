import pandas as pd
import numpy as np


def loadData():
    """ Loads the data and returns the training feautures/labels and test images. """
    print("\n> Loading the data...")

    X_train = np.load('../data/processedData/X_train.npy').astype(np.float32)
    y_train = np.load('../data/processedData/y_train.npy')
    X_test = np.load('../data/processedData/X_test.npy').astype(np.float32)

    return X_train, y_train, X_test


def processRawData():
    """Reads the data from the bare csv files, preprocesses the data and stores it in numpy files for faster access. (Preprocessing and reading from the csv files all the time is slow)"""
    print("\n> Processing the raw data...")

    # -- read data from csv --
    trainData = pd.read_csv('../../data/rawData/trainingData.csv')
    testData = pd.read_csv('../../data/rawData/testData.csv')

    # -- preprocessing --
    # - missing values -
    print("- Checking for missing values...")
    print(trainData.isnull().sum())

    # fill missing values with previous value
    trainData.fillna(method='ffill', inplace=True)
    # remove rows with missing values (would result in having only 2140 images to train on)
    #train_data.reset_index(drop = True,inplace = True)

    print("- Missing values after filling:")
    print(trainData.isnull().any().value_counts())

    print("- Bringing data into shape...")
    # - prepare training features -
    images = trainData['Image'].str.split(" ")
    X_train = np.array(images.to_list(), dtype=np.uint8).reshape(-1, 96, 96, 1)

    # - prepare training labels -
    y_train = trainData.drop('Image', axis=1).to_numpy(dtype=np.float32)

    # - preparing test data -
    tImages = testData['Image'].str.split(" ")
    X_test = np.array(tImages.to_list(), dtype=np.uint8).reshape(-1, 96, 96, 1)

    print("- Storing the data as numpy files...")
    np.save('../../data/processedData/X_train.npy', X_train)
    np.save('../../data/processedData/y_train.npy', y_train)
    np.save('../../data/processedData/X_test.npy', X_test)

    print("- Done!")

# processRawData()
