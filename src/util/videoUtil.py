import cv2 as cv


def listAvailabePorts():
    """Test the first 10 ports and returns the available ports with their output size."""
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
        return int(list(ports.keys())[0])

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
