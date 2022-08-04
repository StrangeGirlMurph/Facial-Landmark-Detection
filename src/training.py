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


def masked_mean_squared_error(y_true, y_pred):
    loss = tf.square(y_true - y_pred)  # Mean Squared Error
    loss = tf.where(y_true != -1., loss, 0.)
    return tf.reduce_mean(loss)


def masked_mean_absolute_error(y_true, y_pred):
    loss = tf.abs(y_true - y_pred)  # Mean Absolute Error
    loss = tf.where(y_true != -1., loss, 0.)
    return tf.reduce_mean(loss)

# Doesnt work yet. Only outputs 0.0
# class Masked_Accuracy(tf.keras.metrics.Accuracy):
#     def __init__(self, name="masked_accuracy", **kwargs):
#         super(Masked_Accuracy, self).__init__(name=name, **kwargs)

#     def update_state(self, y_true, y_pred, sample_weight=None):
#         sample_weight = tf.where(y_true != -1., tf.ones_like(y_true, dtype=tf.float32), 0.)
#         super(Masked_Accuracy, self).update_state(y_true, y_pred, sample_weight)


def masked_accuracy(y_true, y_pred):
    diff = tf.abs(y_true - y_pred)
    diff = tf.where(y_true != -1., diff, 0.)
    passed = tf.math.count_nonzero(diff < 1)  # in margin of one pixel the prediction is counted as correct
    masked_accuracy.inMargin += passed.numpy()
    masked_accuracy.total += tf.size(diff).numpy()
    return masked_accuracy.inMargin / masked_accuracy.total


masked_accuracy.inMargin = 0
masked_accuracy.total = 0
