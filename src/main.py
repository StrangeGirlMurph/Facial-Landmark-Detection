from util.dataUtil import loadData
from util.modelUtil import loadModel, saveModel
from modelStructure import defineModel
from testing import testOnDataset, testOnVideoFile, testOnWebcam
from training import masked_accuracy, trainModel

X_train, y_train, X_test = loadData()

model = loadModel("../models/modelV1")

# model = defineModel()
# trainModel(
#     model, X_train, y_train,
#     epochs=2,
#     batch_size=128,
#     validation_split=0.2,
#     showHistory=True
# )
# saveModel(model, "../models/modelV2")

testOnDataset(model, X_train[30:50], trueValues=y_train[30:50], show=True, save=False, filename="output.png")
# testOnVideoFile(model, videoPath="../data/media/Can You Watch This Without Smiling.mp4")
# testOnWebcam(model)
