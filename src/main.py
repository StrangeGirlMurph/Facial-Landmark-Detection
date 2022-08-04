from util.dataUtil import loadData
from util.modelUtil import loadModel, saveModel
from modelStructure import defineModel
from testing import testOnDataset, testOnVideoFile, testOnWebcam
from training import trainModel

X_train, y_train, X_test = loadData()
numberOfPoints = 20
seed = 30

model = loadModel("../models/modelV2")

# model = defineModel()
# trainModel(
#     model, X_train, y_train,
#     epochs=2,
#     batch_size=128,
#     validation_split=0.2,
#     showHistory=True
# )
# saveModel(model, "../models/modelV2")


testOnDataset(model, X_train[seed:seed+numberOfPoints], trueValues=y_train[seed:seed+numberOfPoints], show=True, save=False, filename="output.png")
# testOnVideoFile(model, videoPath="../data/media/Can You Watch This Without Smiling.mp4")
# testOnWebcam(model)
