from util.dataUtil import loadData
from util.modelUtil import loadModel, saveModel
from modelStructure import defineModel
from testing import testOnDataset, testOnVideoFile, testOnWebcam
from training import trainModel

X_train, y_train, X_test = loadData()
numberOfPoints, seed = 20, 30

model = loadModel("../models/modelV1")
testOnDataset(model, X_train[seed:seed+numberOfPoints], trueValues=y_train[seed:seed+numberOfPoints], show=True, save=True, filename="V1.png")

model = loadModel("../models/modelV2")
testOnDataset(model, X_train[seed:seed+numberOfPoints], trueValues=y_train[seed:seed+numberOfPoints], show=True, save=True, filename="V2.png")

# testOnVideoFile(model, videoPath="../data/media/Can You Watch This Without Smiling.mp4")
# testOnWebcam(model)
