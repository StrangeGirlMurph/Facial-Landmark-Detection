from testing import testOnDataset, testOnVideoFile, testOnWebcam
from util.dataUtil import loadAugmentedData, loadData
from util.modelUtil import loadModel

# X, y = loadData()
# numberOfPoints, seed = 20, 30
# model = loadModel("../models/modelV4")
# testOnDataset(model, X[seed:seed+numberOfPoints], trueValues=y[seed:seed+numberOfPoints], show=True, save=False, filename="V4.png")

model = loadModel("../models/modelV4")
testOnVideoFile(model, videoPath="../data/raw300VW/224/vid.avi", minSize=0.3)

# model = loadModel("../models/modelV4")
# testOnWebcam(model)
