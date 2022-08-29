import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import os


def showImagePyplot(image, x, y, cmap="gray"):
    """Shows the image with the points of the x,y coordinates."""
    plt.imshow(image, cmap=cmap)
    plt.scatter(x, y, c='b', marker='.')
    plt.show()


def mapPointsFromSquareToImage(x, y, c, r, s, w, h):
    """Maps the points to fit image size and returns them. (The prediction points have their origin in the bottom left corner. But opencv uses the top left corner. Plus the function takes care of the scaling and offset.)"""
    return (x*s/96 + c, (y*s/96) + r)


def drawPointsInImage(im, x, y):
    """Plots the points given by the x and y coordinates on the image and returns it."""
    size = min(im.shape[:2])//300
    for i, j in zip(x, y):
        cv.circle(im, (int(i), int(j)), size, (255, 0, 0), -1)
    return im


def prepareImageForPrediction(im):
    """Takes an image and returns it ready for prediction (performs: squaring, converting to grayscale and resizing)."""
    return resizeImageToModelSize(grayImage(squareImage(im))).astype(np.float32)


def grayImage(im):
    """Converts the image to grayscale and returns it."""
    return cv.cvtColor(im, cv.COLOR_BGR2GRAY)


def resizeImageToModelSize(im):
    """Resizes the image to 96x96 and returns it."""
    return cv.resize(im, (96, 96), interpolation=cv.INTER_AREA).reshape(-1, 96, 96, 1)


def mirrorImage(im):
    """Mirrors the image horizontaly and returns it."""
    return cv.flip(im, 1)


def drawSquareInImage(im, x, y, s, rgb=(179, 255, 179), thickness=3):
    """Draws a square in the image with the given coordinates (top left corner), width and height and returns it."""
    cv.rectangle(im, (x, y), (x+s, y+s), rgb, thickness)
    return im


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


def violaJonesGetFaceCascade():
    """Returns the face cascade for the Viola Jones algorithm."""
    cv2_base_dir = os.path.dirname(os.path.abspath(cv.__file__))
    return cv.CascadeClassifier(os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml'))


def violaJones(im, face_cascade, minSize=1080):
    """Performs Viola Jones detection and returns the bounding boxes of the faces."""

    faces = face_cascade.detectMultiScale(
        im,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(minSize, minSize),
        flags=cv.CASCADE_SCALE_IMAGE
    )

    return faces
