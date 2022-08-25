from testing import testOnDataset, testOnVideoFile, testOnWebcam
from util.dataUtil import loadAugmentedData, loadData, generateImages
from util.modelUtil import loadModel

X_train, y_train, X_test = loadData()
numberOfPoints, seed = 20, 30

model = loadModel("../models/modelV4")
#testOnDataset(model, X_train[seed:seed+numberOfPoints], trueValues=y_train[seed:seed+numberOfPoints], show=False, save=True, filename="V4.png")

# model = loadModel("../models/modelV3")
#testOnVideoFile(model, videoPath="../data/media/Can You Watch This Without Smiling.mp4")
# testOnWebcam(model)
