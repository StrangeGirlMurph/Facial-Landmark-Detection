import cv2 as cv
from soupsieve import select


def testOnVideo(videoPath, model):
    cap = cv.VideoCapture(videoPath)

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            cv.imshow('frame', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv.destroyAllWindows()


def testOnWebcamInput():
    port = selectPort()

    cap = cv.VideoCapture(port)

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            cv.imshow('frame', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv.destroyAllWindows()


def listPorts():
    """
    Test the ports and returns the available ports .
    """

    ports = {}

    for i in range(10):  # checks the first 10 ports
        camera = cv.VideoCapture(i, cv.CAP_DSHOW)

        if camera.isOpened():
            ret, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if ret:
                ports[i] = (w, h)

    return ports


def selectPort():
    print("Available ports (width, height):")
    for port, size in listPorts().items():
        print(port, size)

    try:
        port = int(input("Enter the camera port: "))
    except:
        print("That's not a valid port... Try again.")
        return selectPort()

    return port


testOnWebcamInput()
