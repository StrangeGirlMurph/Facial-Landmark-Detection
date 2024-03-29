from util.dataUtil import loadData, loadVideoTestData
from util.modelUtil import loadModel
from training import masked_accuracy, masked_mean_absolute_error, masked_mean_squared_error

# Loading the data
X_test, y_test = loadData(includeAugmented=True, percentageOfUncleanData=1)
X_videoTest, y_videoTest = loadVideoTestData()
print("Number of images from the dataset:", len(X_test))
print("Number of video frames:", len(X_videoTest))

# Loading the model
model = loadModel("../models/modelV4")

# Recompile the model to use the custom metrics
model.compile(
    loss=masked_mean_squared_error,
    metrics=[masked_mean_absolute_error, masked_accuracy],
    run_eagerly=True
)

# Evaluating the model on the test data (batches of 32)
print("Results evaluation on the dataset:", model.evaluate(x=X_test, y=y_test))
print("Results evaluation on video frames:", model.evaluate(x=X_videoTest, y=y_videoTest))
