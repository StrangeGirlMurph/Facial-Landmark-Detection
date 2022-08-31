from testing import testOnDataset, testOnVideoFile, testOnWebcam
from util.dataUtil import loadData
from util.modelUtil import loadModel

# - Option 1: Test on the dataset
# X, y = loadData()
# numberOfPoints, seed = 20, 30
# model = loadModel("../models/modelV4")
# testOnDataset(model, X[seed:seed+numberOfPoints], trueValues=y[seed:seed+numberOfPoints], show=True, save=False, filename="V4.png")

# - Option 2: Test on a video file
model = loadModel("../models/modelV4")
testOnVideoFile(model, videoPath="../data/raw300VW/224/vid.avi", minFaceSize=0.3)

# - Option 3: Test on the webcam
# model = loadModel("../models/modelV4")
# testOnWebcam(model)
