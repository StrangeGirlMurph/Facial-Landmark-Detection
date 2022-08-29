from testing import testOnDataset, testOnVideoFile, testOnWebcam
from util.dataUtil import loadAugmentedData, loadData
from util.modelUtil import loadModel

X_train, y_train = loadData()
numberOfPoints, seed = 20, 30


model = loadModel("../models/modelV4")

# model = loadModel("../models/modelV3")
testOnVideoFile(model, videoPath="../data/testVideos/406.avi", minSize=0.3)
# testOnWebcam(model)
