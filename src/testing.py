import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from util.imageUtil import drawMaxSquareInImage, drawPointsInImage, drawSquareInImage, grayImage, mapPointsFromSquareToImage, resizeImageToModelSize, violaJones, violaJonesGetFaceCascade
from util.videoUtil import selectPort, mirrorImage, prepareImageForPrediction


def testOnDataset(model, data, trueValues=None, show=True, save=False, filename="testOutput.png"):
    """Tests the model on the data and shows result."""
    print("\n> Testing the model...")

    X, Y = predictOnImages(model, data)
    fig = plt.figure(figsize=(28, 20))

    l = len(data)
    i = 1

    for im, x, y in zip(data, X, Y):
        axis = fig.add_subplot(int(np.ceil(l/5)), 5, i)
        axis.imshow(im.reshape(96, 96), cmap='gray')
        plt.scatter(x, y, marker='s', s=8, c="dodgerblue", cmap="tab20")
        if trueValues is not None:
            plt.scatter(trueValues[i-1][0::2], trueValues[i-1][1::2], marker='s', s=8, c="lime", cmap="tab20")
        i += 1

    plt.tight_layout()

    if save:
        plt.savefig(f'../output/{filename}', dpi=300)
    if show:
        plt.show()


def testOnVideoFile(model, videoPath="..\data\media\Can You Watch This Without Smiling.mp4", minSize=0.4):
    """Tests the model on a video file. (Minsize is the minimum size of the face in the image in %)"""
    print("\n> Testing the model on a video...")

    videoLoop(model, videoPath, "Video-Feed", False, minSize)


def testOnWebcam(model):
    """Tests the model on the webcam."""
    print("\n> Testing the model on the webcam feed...")

    port = selectPort()
    videoLoop(model, port, "Webcam-Feed", True)


def videoLoop(model, inp, windowName, mirrored, minSize=0.4):
    """Loops through the video and shows the predictions."""
    print("- Starting the video loop...")
    print("- You can close the window by pressing 'q'.")
    print("- Press 's' to switch between the view modes and 'f' to toggle the frame around the face.")
    print("- Use '+' and '-' to change the size of the window.")
    print("- And press space to pause the feed.")

    # Window
    cap = cv.VideoCapture(inp)
    cv.namedWindow(windowName, cv.WINDOW_KEEPRATIO)
    frameCount = 0

    # Face recognition
    c, r, s = 0, 0, 0  # face location (column, row, sideLength)
    faceCascade = violaJonesGetFaceCascade()
    faceRecognitionFrames = 10  # every nth frame is used for face recognition
    faceRecognitionCountdown = 5  # number of frames to wait before old data is discarded
    countdown = faceRecognitionCountdown
    faceFound = False
    showFrame = True

    # Window size
    scale = 1  # scale of the window
    SCALE_STEP = 0.1  # step size of the window scale
    DEFAULT_PRED_WINDOW_SIZE = 384  # default size of the prediction window
    FEED_WIDTH, FEED_HEIGHT = int(cap.get(3)), int(cap.get(4))
    print(f"- Feed size: {FEED_WIDTH}x{FEED_HEIGHT}")
    DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT = FEED_WIDTH, FEED_HEIGHT
    if min(FEED_WIDTH, FEED_HEIGHT) >= 1080:
        DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT = FEED_WIDTH//2, FEED_HEIGHT//2
    cv.resizeWindow(windowName, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)

    showPredImage = False  # show the 96x96 image that gets predicted on
    pause = False

    while(cap.isOpened()):
        if not pause:
            rv, frame = cap.read()  # BGR
            if not rv:
                break

            # mirroring if webcam input
            if mirrored:
                frame = mirrorImage(frame)

            im = grayImage(frame)

            if frameCount % faceRecognitionFrames == 0:
                faces = violaJones(im, faceCascade, int(min(FEED_WIDTH, FEED_HEIGHT)*minSize))
                if len(faces) != 0:
                    faceFound = True
                    countdown = faceRecognitionCountdown
                    c, r, s1, s2 = faces[0]
                    s = max(s1, s2)
                else:
                    if (countdown == 0):
                        faceFound = False
                    else:
                        countdown -= 1

            if showPredImage:
                im = resizeImageToModelSize(im[r:r+s, c:c+s])
                if faceFound:
                    x, y = predictOnImage(model, im)
                    frame = drawPointsInImage(im[0], x, y)
                else:
                    frame = im[0]
            else:
                if faceFound:
                    im = resizeImageToModelSize(im[r:r+s, c:c+s])
                    x, y = predictOnImage(model, im)
                    x, y = mapPointsFromSquareToImage(x, y, c, r, s, FEED_WIDTH, FEED_HEIGHT)
                    frame = drawPointsInImage(frame, x, y)
                    if showFrame:
                        frame = drawSquareInImage(frame, c, r, s)

            cv.imshow(windowName, frame)

        # input handling
        key = cv.waitKey(1) & 0xFF
        if key == ord('s') or key == ord('+') or key == ord('-'):
            # change window
            if key == ord('+'):
                scale += SCALE_STEP
            elif key == ord('-'):
                scale -= SCALE_STEP
            else:
                showPredImage = not showPredImage

            if showPredImage:
                cv.resizeWindow(windowName, int(DEFAULT_PRED_WINDOW_SIZE * scale), int(DEFAULT_PRED_WINDOW_SIZE * scale))
            else:
                cv.resizeWindow(windowName, int(DEFAULT_WINDOW_WIDTH * scale), int(DEFAULT_WINDOW_HEIGHT * scale))
        elif key == ord('f'):
            showFrame = not showFrame
        elif key == ord('q'):
            # close the window
            break
        elif key == ord(' '):
            # pause the video
            pause = not pause

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
