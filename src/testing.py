import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from util.imageUtil import drawMaxSquareInImage, drawPointsInImage, drawSquareInImage, grayImage, mapPointsFromSquareToImage, resizeImageToModelSize, violaJones, violaJonesGetFaceCascade
from util.videoUtil import selectPort, mirrorImage, prepareImageForPrediction


def testOnDataset(model, data, trueValues=None, show=False, save=True, filename="testOutput.png"):
    """Tests the model on the data and shows result."""
    print("\n> Testing the model...")

    X, Y = predictOnImages(model, data)
    fig = plt.figure(figsize=(28, 20))

    l = len(data)
    i = 1

    for im, x, y in zip(data, X, Y):
        axis = fig.add_subplot(int(np.ceil(l/5)), 5, i)
        axis.imshow(im, cmap='gray')
        plt.scatter(x, y, c='b', marker='.')
        if trueValues is not None:
            plt.scatter(trueValues[i-1][0::2], trueValues[i-1][1::2], c='r', marker='.')
        i += 1

    plt.tight_layout()

    if save:
        plt.savefig(f'../out/{filename}', dpi=300)
    if show:
        plt.show()


def testOnVideoFile(model, videoPath="..\data\media\Can You Watch This Without Smiling.mp4"):
    """Tests the model on a video file."""
    print("\n> Testing the model on a video...")

    videoLoop(model, videoPath, "Video-Feed", False)


def testOnWebcam(model):
    """Tests the model on the webcam."""
    print("\n> Testing the model on the webcam feed...")

    port = selectPort()
    videoLoop(model, port, "Webcam-Feed", True)


def videoLoop(model, inp, windowName, mirrored):
    """Loops through the video and shows the predictions."""
    print("- Starting the video loop...")

    faceCascade = violaJonesGetFaceCascade()
    c, r, s = 0, 0, 0  # face location (column, row, sideLength)

    cap = cv.VideoCapture(inp)
    w, h = cap.get(3), cap.get(4)
    cv.namedWindow(windowName, cv.WINDOW_KEEPRATIO)

    print("- You can close the window by pressing 'q'")
    frameCount = 0
    while(cap.isOpened()):
        rv, frame = cap.read()  # BGR
        if rv == True:
            if mirrored:
                frame = mirrorImage(frame)

            im = grayImage(frame)
            if frameCount % 10 == 0:
                # every 10 frames, detect faces and draw them
                faces = violaJones(im, faceCascade)
                if len(faces) != 0:
                    c, r, s1, s2 = faces[0]
                    s = max(s1, s2)

            im = resizeImageToModelSize(im[r:r+s, c:c+s])
            x, y = predictOnImage(model, im)
            x, y = mapPointsFromSquareToImage(x, y, c, r, s, w, h)

            frame = drawPointsInImage(frame, x, y)
            frame = drawSquareInImage(frame, c, r, s)
            cv.imshow(windowName, frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv.destroyAllWindows()


def predictOnImage(model, im):
    """Predicts on a single image and returns the coordinates of the predicted points."""
    pred = model.predict(im, verbose=0)
    return (pred[0][0::2], pred[0][1::2])


def predictOnImages(model, ims):
    """Predicts on multiple images and returns the list of coordinates of the predicted points."""
    pred = model.predict(ims)
    return ([i[0::2] for i in pred], [i[1::2] for i in pred])
