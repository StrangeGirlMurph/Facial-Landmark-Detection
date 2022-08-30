from util.augmentationUtil import rotate, horizontalFlip, cropAndPad, perspective, brightnessAndContrast
from util.imageUtil import violaJones, violaJonesGetFaceCascade, grayImage, resizeImageToModelSize
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2 as cv
import random


def loadData(includeAugmented=True, percentageOfUncleanData=1, includeTestData=False):
    """Loads the data and returns the training feautures/labels and test images. Also lets you choose how much of the unclean data you want to include."""
    print("\n> Loading the data...")

    X_train = np.load('../data/processedData/X_train.npy').astype(np.float32)
    y_train = np.load('../data/processedData/y_train.npy').astype(np.float32)
    X_test = np.load('../data/processedData/X_test.npy').astype(np.float32)

    a, _ = np.where(y_train == -1)
    a = np.unique(a)
    i = np.random.choice(a, int(len(a)*(1-percentageOfUncleanData)), replace=False)
    X_train = np.delete(X_train, i, axis=0)
    y_train = np.delete(y_train, i, axis=0)

    if includeAugmented:
        X_augmented, y_augmented = loadAugmentedData()
        X_train = np.concatenate((X_train, X_augmented))
        y_train = np.concatenate((y_train, y_augmented))

    if includeTestData:
        return X_train, y_train, X_test
    return X_train, y_train


def loadAugmentedData():
    """Loads the augmented data from the numpy files."""
    return np.load('../data/processedData/X_augmented.npy').astype(np.float32), np.load('../data/processedData/y_augmented.npy').astype(np.float32)


def loadVideoTestData():
    """Loads the video data from the numpy files."""
    return np.load("../data/processed300VW/X_test.npy").astype(np.float32), np.load("../data/processed300VW/y_test.npy").astype(np.float32)


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


def processRawTestVideos(videos, framesPerVideo=.1):
    """Processes the given videos from the 300VW dataset and stores them in npy files."""
    print("\n> Processing raw test videos...")
    X, y = np.empty((0, 96, 96, 1), dtype=np.uint8), np.empty((0, 30), dtype=np.float16)

    for video in videos:
        Xn, yn = processRawTestVideo(video, framesPerVideo)
        X = np.concatenate((X, Xn), axis=0)
        y = np.concatenate((y, yn), axis=0)

    print("Features: ", X.shape, X.dtype)
    print("Labels: ", y.shape, y.dtype)
    np.save("../data/processed300VW/X_test.npy", X)
    np.save("../data/processed300VW/y_test.npy", y)
    print("- Done!")


def processRawTestVideo(num, frames=0.1):
    """Processes the raw test video and stores it in numpy files for faster access. Frames is either a percentage or a total number of frames."""
    print(f"- Processing video number {num}...")
    cap = cv.VideoCapture(f"../data/raw300VW/{num:03d}/vid.avi")
    if not cap.isOpened():
        raise Exception("Could not open video")
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv.CAP_PROP_FPS))
    totalFrames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    print("(w,h,fps,frames):", width, height, fps, totalFrames)

    if frames <= 1 or frames > totalFrames:
        size = int(frames * totalFrames)
    else:
        size = int(frames)
    portion = np.random.choice(np.arange(1, totalFrames+1), size=size, replace=False)

    minSize = 0.4
    faceCascade = violaJonesGetFaceCascade()
    outX = np.empty((0, 96, 96, 1), dtype=np.uint8)
    outY = np.empty((0, 30), dtype=np.float16)

    for i in tqdm(range(totalFrames)):
        if not cap.isOpened():
            break

        rv, frame = cap.read()

        if not rv:
            break

        if i in portion:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            points = np.loadtxt(f"../data/raw300VW/{num:03d}/annot/{i:06d}.pts", comments=("version:", "n_points:", "{", "}")).astype(np.float16)

            y = np.empty((15, 2), dtype=np.float16)
            indices = [0, 0, 42, 45, 39, 36, 22, 26, 21, 17, 30, 54, 48, 0, 0]
            y = points[indices]
            y[0] = calculatePointInBetween(calculatePointInBetween(points[46], points[47]), calculatePointInBetween(points[43], points[44]))
            y[1] = calculatePointInBetween(calculatePointInBetween(points[40], points[41]), calculatePointInBetween(points[37], points[38]))
            y[13] = calculatePointInBetween(points[51], points[62])
            y[14] = calculatePointInBetween(points[57], points[66])

            im = grayImage(frame)
            faces = violaJones(im, faceCascade, int(min(width, height)*minSize))
            if len(faces) != 0:
                c, r, s1, s2 = faces[0]
                s = max(s1, s2)
            else:
                continue

            y[:, 0] = (y[:, 0] - c) * 96 / s
            y[:, 1] = (y[:, 1] - r) * 96 / s

            im = resizeImageToModelSize(im[r:r+s, c:c+s])
            outX = np.append(outX, im, axis=0)
            outY = np.append(outY, [y.flatten()], axis=0)

    cap.release()
    cv.destroyAllWindows()

    return outX, outY


def calculatePointInBetween(p1, p2):
    """Calculates the point in between two x,y points."""
    return [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]


def printDatasetOverview():
    """Prints some basic information about the dataset."""
    print("\n> Printing dataset overview...")
    data = pd.read_csv('../data/rawData/trainingData.csv')

    print("Head:\n", data.head().T[0])
    print("Info:\n", data.info())
