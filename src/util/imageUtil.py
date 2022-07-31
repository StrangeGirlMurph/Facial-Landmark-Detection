import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np


def showImage(image, x, y, cmap="gray"):
    """Shows the image of size 96x96 with the points of the x,y coordinates."""
    plt.imshow(image, cmap=cmap)
    plt.scatter(x, y, c='b', marker='.')
    plt.show()


def mapPointsToImageSize(x, y, w, h):
    """Maps the points to fit image size and returns them. (The prediction points have their origin in the bottom left corner. But opencv uses the top left corner. Plus the function takes care of the scaling and offset.)"""
    mid = np.array([w//2, h//2])
    sideLength = min(w, h)
    zero = mid - [sideLength//2, sideLength//2]
    scaling = sideLength/96
    return (x*scaling + zero[0], (y*scaling) + zero[1])


def drawPointsInImage(im, x, y):
    """Plots the points given by the x and y coordinates on the image and returns it."""
    for i, j in zip(x, y):
        cv.circle(im, (int(i), int(j)), 3, (255, 0, 0), -1)
    return im


def prepareImageForPrediction(im):
    """Takes an image and returns it ready for prediction (performs: squaring, converting to grayscale and resizing)."""
    return resizeImage(grayImage(squareImage(im))).astype(np.float32)


def grayImage(im):
    """Converts the image to grayscale and returns it."""
    return cv.cvtColor(im, cv.COLOR_BGR2GRAY)


def resizeImage(im):
    """Resizes the image to 96x96 and returns it."""
    return cv.resize(im, (96, 96), interpolation=cv.INTER_AREA).reshape(-1, 96, 96, 1)


def mirrorImage(img):
    """Mirrors the image horizontaly and returns it."""
    return cv.flip(img, 1)


def drawMaxSquareInImage(im):
    """"Draws the maximum possible square in the center of the image and returns it."""
    h, w = im.shape[:2]
    ds = min(h, w)//2
    h, w = h//2, w//2   # center
    cv.rectangle(im, (w-ds, h-ds), (w+ds, h+ds), (0, 0, 0), 1)
    return im


def squareImage(im):
    """Squares the image and returns it. (The size is determind by the minimum of width and height of the input.)"""
    h, w = im.shape[:2]
    ds = min(h, w)//2
    h, w = h//2, w//2   # center
    return im[h-ds:h+ds, w-ds:w+ds]


def violaJones(im):
    """Performs Viola Jones detection and returns the bounding boxes of the faces."""
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(im, 1.3, 5)

    grayscale_image = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
    detected_faces = face_cascade.detectMultiScale(grayscale_image)
    for (column, row, width, height) in detected_faces:
        cv.rectangle(im, (column, row), (column + width, row + height), (0, 255, 0), 4)
