from util.dataUtil import loadData
from util.modelUtil import loadModel
from model import defineModel
from test import testOnDataset, testOnVideoFile, testOnWebcam
from train import trainModel

X_train, y_train, X_test = loadData()

model = loadModel("../models/modelV1")

# model = defineModel()
# trainModel(
#     model, X_train, y_train,
#     epochs=20,
#     batch_size=256,
#     validation_split=0.2,
#     save=(True, "modelV1")
# )
# saveModel(model, "../models/modelV2")

# testOnDataset(model, X_test[30:50])
# testOnWebcam(model)
testOnVideoFile(model, videoPath="../data/media/Can You Watch This Without Smiling.mp4")
