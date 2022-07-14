def predictOnImage(model, image):
    """Predicts on the input image and returns a list of x and y coordinates. """

    pred = model.predict(image)
    return pred[0::2], pred[1::2]


def loadModel(modelPath):
    """ Loads the model from the given path."""

    pass
