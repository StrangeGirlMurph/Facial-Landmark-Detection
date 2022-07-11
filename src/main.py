from data_loading import loadData
from architecture import defineModel
from test import testModel
from train import trainModel

X_train, y_train, X_test = loadData()
model = defineModel()

trainModel(
    model, X_train, y_train,
    epochs=1,
    batch_size=128,
    validation_split=0.2,
    save=(True, "modelV1")
)

testModel(model, X_test)
