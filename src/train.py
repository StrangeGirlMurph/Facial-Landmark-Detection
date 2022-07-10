

def trainModel(model, X_train, y_train, epochs=50, batch_size=256, validation_split=0.2, save=(True, "model1")):
    """ 
    Trains the model and can save it under the given name.
    """

    print("\n> Training the model...")
    model.compile(
        optimizer='adam',           # stochastic gradient descent
        loss='mean_squared_error',  # mean_squared_error
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
        model.save("../models/" + save[1])
