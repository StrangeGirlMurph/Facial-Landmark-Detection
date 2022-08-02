from util.dataUtil import loadData
from util.modelUtil import loadModel, saveModel
from modelStructure import defineModel
from testing import testOnDataset, testOnVideoFile, testOnWebcam
from training import trainModel

X_train, y_train, X_test = loadData()

model = loadModel("../models/modelV1")

# model = defineModel()
# trainModel(
#     model, X_train, y_train,
#     epochs=20,
#     batch_size=256,
#     validation_split=0.2,
#     showHistory=True
# )
# saveModel(model, "../models/modelV2")

testOnDataset(model, X_test[30:50], show=True, save=True)
# testOnVideoFile(model, videoPath="../data/media/Can You Watch This Without Smiling.mp4")
# testOnWebcam(model)
