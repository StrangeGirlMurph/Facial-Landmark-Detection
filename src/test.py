import matplotlib.pyplot as plt
from util.imageUtil import *
from util.videoUtil import *


def testOnDataset(model, data):
    """Tests the model on the data and shows result."""
    print("\n> Testing the model...")

    X, Y = predictOnImages(model, data)
    for im, x, y in zip(data, X, Y):
        showImage(im, x, y)


def testOnVideo(model=2, videoPath=""):
    """Tests the model on a video input. Either a path to a video or direct camera input."""
    print("\n> Testing the model on a video...")

    if not videoPath:
        port = selectPort()
        isCameraInput = True
    else:
        port = videoPath
        isCameraInput = False

    cap = cv.VideoCapture(port)

    print("- You can close the window by pressing 'q'")

    while(cap.isOpened()):
        rv, frame = cap.read()  # BGR

        if rv == True:
            if isCameraInput:
                frame = mirrorImage(frame)

            im = prepareImageForPrediction(frame)
            x, y = predictOnImage(model, im)
            frame = plotPointsOnImage(frame, x, y)

            cv.imshow("Video-Feed", frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv.destroyAllWindows()


def predictOnImage(model, im):
    """Predicts on a single image and returns the coordinates of the predicted points."""
    pred = model.predict(im)
    return (pred[0::2], pred[1::2])


def predictOnImages(model, ims):
    """Predicts on multiple images and returns the list of coordinates of the predicted points."""
    pred = model.predict(ims)
    return ([i[0::2] for i in pred], [i[1::2] for i in pred])


testOnVideo()
