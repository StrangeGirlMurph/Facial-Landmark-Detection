from util.modelUtil import showTrainingHistory
import tensorflow as tf


def trainModel(model, X_train, y_train, epochs=50, batch_size=256, validation_split=0.2, showHistory=True):
    """Compiles and fits the model inplace plus saves it if specified under the given name. """
    print("\n> Training the model...")
    print(f"> Epochs: {epochs}, Batch size: {batch_size}, Validation split: {validation_split}")

    model.compile(
        optimizer='adam',   # stochastic gradient descent
        loss=masked_mean_squared_error,
        metrics=[masked_mean_absolute_error, masked_accuracy],
        run_eagerly=True    # custom metrics
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,                      # number of epochs: 50
        batch_size=batch_size,              # batch size of 256
        validation_split=validation_split   # 20% of data for validation
    )

    if showHistory:
        showTrainingHistory(history)


def filter_mask(y_true, y_pred):
    mask = tf.math.logical_not(tf.math.equal(y_true, tf.constant(-1.)))
    y_true = tf.ragged.boolean_mask(y_true, mask)
    y_pred = tf.ragged.boolean_mask(y_pred, mask)
    return y_true, y_pred


def masked_mean_squared_error(y_true, y_pred):
    y_true, y_pred = filter_mask(y_true, y_pred)
    return tf.keras.losses.mean_squared_error(y_true, y_pred)


def masked_mean_absolute_error(y_true, y_pred):
    y_true, y_pred = filter_mask(y_true, y_pred)
    return tf.keras.losses.mean_absolute_error(y_true, y_pred)


def masked_accuracy(y_true, y_pred):
    y_true, y_pred = filter_mask(y_true, y_pred)
    diff = tf.reshape(tf.abs(y_true - y_pred), [-1])
    passed = tf.math.count_nonzero(diff < 2)  # in margin of one pixel the prediction is counted as correct
    masked_accuracy.inMargin += passed.numpy()
    masked_accuracy.total += tf.size(diff).numpy()
    return masked_accuracy.inMargin / masked_accuracy.total


masked_accuracy.inMargin = 0
masked_accuracy.total = 0
