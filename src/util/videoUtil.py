from util.imageUtil import prepareImageForPrediction, mirrorImage
import matplotlib.pyplot as plt
import cv2 as cv


def listAvailabePorts():
    """Test the first 10 ports and returns the available ports ones with their output size."""
    ports = {}

    for i in range(10):
        # checks the first 10 ports (0-9)
        camera = cv.VideoCapture(i, cv.CAP_DSHOW)

        if camera.isOpened():
            rv, im = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if rv:
                ports[i] = (w, h)

    return ports


def selectPort():
    """Lets the user select a port and returns the port number."""
    ports = listAvailabePorts()

    if not ports:
        print("- No ports found... Trying port 0.")
        return 0

    if len(ports) == 1:
        print("- Only one port available:", list(ports.keys())[0])
        return list(ports.keys())[0]

    print("Available ports (width, height):")
    for port, size in ports.items():
        print("Port:", port, "-", size)

    try:
        port = int(input("Enter the camera port: "))
    except:
        pass

    if port != 0 and port not in ports.keys():
        print("That's not a valid port... Try again.")
        return selectPort()

    return port


def showVideoFeed(videoPath="", showReadyForPred=False):
    """Shows you the video input. Either a path to a video or direct camera input."""
    print("\n> Testing the video feed...")

    if not videoPath:
        port = selectPort()
        isCameraInput = True
    else:
        port = videoPath
        isCameraInput = False

    cap = cv.VideoCapture(port)
    cv.namedWindow("Video-Feed")
    cv.setWindowProperty('Video-Feed', 1, cv.WINDOW_NORMAL)

    if showReadyForPred:
        cv.resizeWindow("Video-Feed", 480, 480)

    print("- You can close the window by pressing 'q'")

    while(cap.isOpened()):
        rv, frame = cap.read()  # BGR

        if rv == True:
            if isCameraInput:
                frame = mirrorImage(frame)

            if showReadyForPred:
                frame = prepareImageForPrediction(frame)

            cv.imshow("Video-Feed", frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv.destroyAllWindows()
