import matplotlib.pyplot as plt
from util.imageUtil import showImage


def testModel(model, X_test):
    """ Tests the model on the test data and shows a result. """
    print("\n> Testing the model...")

    pred = model.predict(X_test)
    showImage(X_test[30], pred[30])
