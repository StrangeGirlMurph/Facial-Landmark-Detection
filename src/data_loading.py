import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from util.imageUtil import showImage


def loadData():
    """ Loads the data and returns the training feautures/labels and test images. """
    print("\n> Loading the data...")

    # -- loading data --
    Train_Dir = '../data/trainingData.csv'
    Test_Dir = '../data/testData.csv'
    train_data = pd.read_csv(Train_Dir)
    test_data = pd.read_csv(Test_Dir)
    os.listdir('../data')

    # print(train_data.head().T)

    # -- preprocessing --
    # - missing values -
    print("- Checking for missing values...")
    # print(train_data.isnull().sum())

    # fill missing values with previous value
    train_data.fillna(method='ffill', inplace=True)
    # remove rows with missing values (would result in having only 2140 images to train on)
    #train_data.reset_index(drop = True,inplace = True)

    print("- Missing values after filling:")
    print(train_data.isnull().any().value_counts())

    print("- Preprocessing...")
    # - prepare features -
    images = train_data['Image'].str.split(" ")
    X_train = np.array(images.to_list(), dtype="float").reshape(-1, 96, 96, 1)

    # - prepare labels -
    y_train = train_data.drop('Image', axis=1).to_numpy(dtype="float")

    # - preparing test data -
    timages = test_data['Image'].str.split(" ")
    X_test = np.array(timages.to_list(), dtype="float").reshape(-1, 96, 96, 1)

    showImage(X_train[30], y_train[30])

    return X_train, y_train, X_test
