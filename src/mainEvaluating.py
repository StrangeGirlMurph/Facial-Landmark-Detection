from util.dataUtil import loadData, loadVideoTestData
from util.modelUtil import loadModel
from training import masked_accuracy, masked_mean_absolute_error, masked_mean_squared_error

# Loading the data
X_test, y_test = loadData(includeAugmented=True, percentageOfUncleanData=0)
X_videoTest, y_videoTest = loadVideoTestData()

# Loading the model
model = loadModel("../models/modelV4")

# recompile the model to use the custom metrics
model.compile(
    loss=masked_mean_squared_error,
    metrics=[masked_mean_absolute_error, masked_accuracy],
    run_eagerly=True
)

# Evaluating the model on the test data (batches of 32)
print(model.evaluate(x=X_test, y=y_test))
print(model.evaluate(x=X_videoTest, y=y_videoTest))
