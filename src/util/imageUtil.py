import matplotlib.pyplot as plt
import cv2 as cv


def showImage(image, x, y, cmap="gray"):
    """Shows the image of size 96x96 with the points of the x,y coordinates."""
    plt.imshow(image, cmap=cmap)
    plt.scatter(x, y, c='b', marker='.')
    plt.show()


def plotPointsOnImage(im, x, y):
    """Plots the points given by the x and y coordinates on the image and returns it."""
    for i, j in zip(x, y):
        cv.circle(im, (i, j), 1, (0, 0, 255), -1)
    return im


def prepareImageForPrediction(im):
    """Takes an image and returns it ready for prediction (performs: squaring, converting to grayscale and resizing)."""
    return resizeImage(grayImage(squareImage(im)))


def grayImage(im):
    """Converts the image to grayscale and returns it."""
    return cv.cvtColor(im, cv.COLOR_BGR2GRAY)


def resizeImage(im):
    """Resizes the image to 96x96 and returns it."""
    return cv.resize(im, (96, 96), interpolation=cv.INTER_AREA)


def mirrorImage(img):
    """Mirrors the image horizontaly and returns it."""
    return cv.flip(img, 1)


def squareImage(img):
    """Squares the image and returns it. (The size is determind by the minimum of width and height of the input.)"""
    h, w = img.shape[:2]
    ds = min(h, w)//2
    h, w = h//2, w//2   # center
    return img[h-ds:h+ds, w-ds:w+ds]
