from util.modelUtil import showTrainingHistory
import tensorflow as tf


def trainModel(model, X_train, y_train, epochs=50, batch_size=256, validation_split=0.2, showHistory=True):
    """Compiles and fits the model inplace."""
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


def filter_mask(y_true):
    """Returns a mask of the same shape as y_true, where each element is 1 if the corresponding element in y_true is -1 and 0 otherwise."""
    return tf.math.logical_not(tf.math.equal(y_true, tf.constant(-1.)))


def masked_mean_squared_error(y_true, y_pred):
    """Returns the mean squared error between y_true and y_pred and masking the values where y_true is -1."""
    mask = filter_mask(y_true)
    loss = tf.square(tf.abs(y_true - y_pred))
    return tf.reduce_mean(tf.ragged.boolean_mask(loss, mask), 1)


def masked_mean_absolute_error(y_true, y_pred):
    """Returns the mean absolute error between y_true and y_pred and masking the values where y_true is -1."""
    mask = filter_mask(y_true)
    loss = tf.abs(y_true - y_pred)
    return tf.reduce_mean(tf.ragged.boolean_mask(loss, mask), 1)


def masked_accuracy(y_true, y_pred):
    """Returns the overall accuracy with the accuracy between y_true and y_pred added and masking the values where y_true is -1."""
    mask = filter_mask(y_true)
    diff = tf.abs(y_true - y_pred)
    diff = tf.boolean_mask(diff, mask)
    # in margin of 2 pixel the prediction is counted as correct
    passed = tf.math.count_nonzero(tf.less(diff, 2))
    masked_accuracy.inMargin += passed.numpy()
    masked_accuracy.total += tf.size(diff).numpy()
    return masked_accuracy.inMargin / masked_accuracy.total


masked_accuracy.inMargin = 0
masked_accuracy.total = 0
