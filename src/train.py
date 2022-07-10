from architecture import model
from data_loading import X_train, y_train


print("\n> Training the model...")
model.compile(
    optimizer='adam',           # stochastic gradient descent
    loss='mean_squared_error',  # mean_squared_error
    metrics=['mae']             # mean absolute error
)

model.fit(
    X_train,
    y_train,
    epochs=1,               # number of epochs: 50
    batch_size=128,         # batch size of 256
    validation_split=0.2    # 20% of data for validation
)
