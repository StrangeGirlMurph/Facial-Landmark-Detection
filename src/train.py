from util.modelUtil import saveModel


def trainModel(model, X_train, y_train, epochs=50, batch_size=256, validation_split=0.2, save=(True, "modelV1")):
    """Compiles and fits the model inplace plus saves it if specified under the given name. """
    print("\n> Training the model...")
    print(f"> Epochs: {epochs}, Batch size: {batch_size}, Validation split: {validation_split}")

    model.compile(
        optimizer='adam',           # stochastic gradient descent
        loss='mean_squared_error',  # mean square error
        metrics=['mae']             # mean absolute error
    )

    model.fit(
        X_train,
        y_train,
        epochs=epochs,                      # number of epochs: 50
        batch_size=batch_size,              # batch size of 256
        validation_split=validation_split   # 20% of data for validation
    )

    if save[0]:
        print("- Saving the model...")
        saveModel(model, "../models/" + save[1])


def predictOnImage(model, image):
    """Predicts on the input image and returns a list of x and y coordinates. """
    pred = model.predict(image)
    return pred[0::2], pred[1::2]
