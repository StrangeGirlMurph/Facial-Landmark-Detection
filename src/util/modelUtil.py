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
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    # summarize history for mean_absolute_error
    ax1.plot(history.history['mae'])
    ax1.plot(history.history['val_mae'])
    ax1.set_title('Mean Absolute Error vs Epoch')
    ax1.set_ylabel('Mean Absolute Error')
    ax1.set_xlabel('Epochs')
    ax1.legend(['train', 'validation'], loc='upper right')
    # summarize history for accuracy
    ax2.plot(history.history['acc'])
    ax2.plot(history.history['val_acc'])
    ax2.set_title('Accuracy vs Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.legend(['train', 'validation'], loc='upper left')
    # summarize history for loss
    ax3.plot(history.history['loss'])
    ax3.plot(history.history['val_loss'])
    ax3.set_title('Loss vs Epoch')
    ax3.set_ylabel('Loss')
    ax3.set_xlabel('Epochs')
    ax3.legend(['train', 'validation'], loc='upper left')

    plt.tight_layout()
    plt.show()
