import keras


def saveModel(model=None, path="../../models/modelV1"):
    """Saves the model to the given path."""
    print("\n> Saving the model...")
    if model is None:
        raise Exception("No model to save.")
    model.save(path)
    print("Model saved to: " + path)


def loadModel(modelPath):
    """ Loads the model from the given path."""
    print("\n> Loading the model...")
    return keras.models.load_model(modelPath)


def summarizeModel(model):
    """Prints a summary of the model."""
    model.summary()


def visualizeModel(model, path='../models/modelImage.png'):
    """Creates an image which summarizes the model and visualizes the layers."""
    keras.utils.plot_model(model, to_file=path, show_shapes=True)
