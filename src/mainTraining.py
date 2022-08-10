from util.dataUtil import loadData
from util.modelUtil import saveModel
from modelStructure import defineModel
from training import trainModel

X_train, y_train, X_test = loadData(includeAugmented=True)
model = defineModel()

trainModel(
    model, X_train, y_train,
    epochs=100,
    batch_size=256,
    validation_split=0.2,
    showHistory=True
)

saveModel(model, "../models/modelV3")
