from util.imageUtil import drawPointsInImage, drawSquareInImage, grayImage, mapPointsFromSquareToImage, resizeImageToModelSize, violaJones, violaJonesGetFaceCascade, mirrorImage
from util.videoUtil import selectPort
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


def testOnDataset(model, data, trueValues=None, show=True, save=False, filename="testOutput.png"):
    """Tests the model on the data and shows result."""
    print("\n> Testing the model...")

    X, Y = predictOnImages(model, data)

    fig = plt.figure(figsize=(28, 20))
    plt.tight_layout()

    # To handle the rows and columns of the plot
    l = len(data)
    i = 0

    for im, x, y in zip(data, X, Y):
        axis = fig.add_subplot(int(np.ceil(l/5)), 5, i+1)

        axis.imshow(im.reshape(96, 96), cmap='gray')
        plt.scatter(x, y, marker='s', s=8, c="dodgerblue", cmap="tab20")

        if trueValues is not None:
            plt.scatter(trueValues[i][0::2], trueValues[i][1::2], marker='s', s=8, c="lime", cmap="tab20")

        i += 1

    if save:
        plt.savefig(f'../output/{filename}', dpi=300)
    if show:
        plt.show()


def testOnVideoFile(model, videoPath="../data/raw300VW/406/vid.avi", minFaceSize=0.4):
    """Tests the model on a video file. (minFaceSize is the minimum size to detect a face in the video in %)"""
    print("\n> Testing the model on a video...")

    videoLoop(model, videoPath, "Video-Feed", False, minFaceSize)


def testOnWebcam(model, minFaceSize=0.4):
    """Tests the model on the webcam. (minFaceSize is the minimum size to detect a face in the video in %)"""
    print("\n> Testing the model on the webcam feed...")

    port = selectPort()
    videoLoop(model, port, "Webcam-Feed", True, minFaceSize)


def videoLoop(model, inp, windowName, mirrored, minFaceSize=0.4):
    """Loops through the video and shows the predictions."""
    print("- Starting the video loop...")
    print("- You can close the window by pressing 'q'.")
    print("- And press space to pause the feed.")
    print("- Press 's' to switch between the view modes and 'f' to toggle the frame around the face.")
    print("- Use '+' and '-' to change the size of the window.")

    # Window
    cap = cv.VideoCapture(inp)
    cv.namedWindow(windowName, cv.WINDOW_KEEPRATIO)
    frameCount = 0
    pause = False
    # If true show the 96x96 image that gets predicted on (toggle by pressing 's')
    showPredImage = False

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
    scaleStep = 0.1  # step size of the window scale
    defaultPredWindowSize = 384  # default size of the prediction window
    feedWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    feedHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    print(f"- Feed size: {feedWidth}x{feedHeight}")
    defaultWindowWidth, defaultWindowHeight = feedWidth, feedHeight

    if min(feedWidth, feedHeight) >= 1080:
        defaultWindowWidth, defaultWindowHeight = feedWidth//2, feedHeight//2
    cv.resizeWindow(windowName, defaultWindowWidth, defaultWindowHeight)

    while(cap.isOpened()):
        if not pause:
            rv, frame = cap.read()

            if not rv:
                break

            if mirrored:
                # mirroring the frame if the input is from a webcam
                frame = mirrorImage(frame)

            im = grayImage(frame)

            # Face recognition
            if frameCount % faceRecognitionFrames == 0:
                faces = violaJones(im, faceCascade, int(min(feedWidth, feedHeight)*minFaceSize))
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

            # Prediction and image overlay
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
                    x, y = mapPointsFromSquareToImage(x, y, c, r, s)
                    frame = drawPointsInImage(frame, x, y)
                    if showFrame:
                        frame = drawSquareInImage(frame, c, r, s)

            cv.imshow(windowName, frame)  # show the result

        # input handling
        key = cv.waitKey(1) & 0xFF
        if key == ord('s') or key == ord('+') or key == ord('-'):
            # change window settings
            if key == ord('+'):
                scale += scaleStep
            elif key == ord('-'):
                scale -= scaleStep
            else:
                showPredImage = not showPredImage
            if showPredImage:
                cv.resizeWindow(windowName, int(defaultPredWindowSize * scale), int(defaultPredWindowSize * scale))
            else:
                cv.resizeWindow(windowName, int(defaultWindowWidth * scale), int(defaultWindowHeight * scale))
        elif key == ord('f'):
            showFrame = not showFrame
        elif key == ord('q'):
            break  # close the window
        elif key == ord(' '):
            pause = not pause  # pause the video

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
