import keras
import matplotlib.pyplot as plt


def saveModel(model, path="../../models/modelV1"):
    """Saves the model to the given path."""
    print("\n> Saving the model...")
    model.save(path)
    print("Model saved to: " + path)


def loadModel(modelPath):
    """ Loads the model from the given path."""
    print("\n> Loading the model...")
    return keras.models.load_model(modelPath)


def summarizeModel(model):
    """Prints a summary of the model."""
    print("\n> Model summary:")
    model.summary()


def visualizeModel(model, path='../models/modelImage.png'):
    """Creates an image which summarizes the model and visualizes the layers."""
    keras.utils.plot_model(model, to_file=path, show_shapes=True)
    print("\n> Model visualization saved to: " + path)


def showTrainingHistory(history):
    """Shows the training history of the model."""
    keys = list(history.history.keys())  # for example ['loss', 'accuracy', 'val_loss', 'val_accuracy']
    mid = int(len(keys) / 2)
    fig, axes = plt.subplots(1, mid)

    for i, ax in enumerate(axes):
        ax.plot(history.history[keys[i]])
        ax.plot(history.history[keys[i + mid]])
        ax.set_ylabel(keys[i])
        ax.set_xlabel('Epochs')
        ax.legend(['train', 'validation'], loc='lower right')

    plt.tight_layout()
    plt.show()
