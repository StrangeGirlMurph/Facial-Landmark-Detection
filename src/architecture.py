import keras
from keras.models import Sequential
from keras.layers import Convolution2D, BatchNormalization, Flatten, Dense, Dropout, MaxPool2D, LeakyReLU


def defineModel():
    """ Creates a sequetial model, defines it's architecture and returns it. """
    print("\n> Defining the model...")

    # - fully connected neural network -
    # model = Sequential([
    #     Flatten(input_shape=(96, 96)),
    #     Dense(128, activation="relu"),
    #     Dropout(0.1),
    #     Dense(64, activation="relu"),
    #     Dense(30)
    # ])

    # - convolutional neural network -
    model = Sequential()

    model.add(Convolution2D(32, (3, 3), padding='same', use_bias=False, input_shape=(96, 96, 1)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())

    model.add(Convolution2D(32, (3, 3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, (3, 3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())

    model.add(Convolution2D(64, (3, 3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Convolution2D(96, (3, 3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())

    model.add(Convolution2D(96, (3, 3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Convolution2D(128, (3, 3), padding='same', use_bias=False))
    # model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())

    model.add(Convolution2D(128, (3, 3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Convolution2D(256, (3, 3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())

    model.add(Convolution2D(256, (3, 3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Convolution2D(512, (3, 3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())

    model.add(Convolution2D(512, (3, 3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(30))

    return model


def summarizeModel(model):
    """Prints a summary of the model."""
    model.summary()


def visualizeModel(model, path='../models/modelImage.png'):
    """Creates an image which summarizes the model and visualizes the layers."""
    keras.utils.plot_model(model, to_file=path, show_shapes=True)
