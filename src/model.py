import keras


def predictOnImage(model, image):
    """Predicts on the input image and returns a list of x and y coordinates. """
    pred = model.predict(image)
    return pred[0::2], pred[1::2]
