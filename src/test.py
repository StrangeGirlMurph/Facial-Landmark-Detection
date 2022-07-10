import matplotlib.pyplot as plt


def testModel(model, X_test):
    """ 
    Tests the model on the test data and shows a result.
    """

    print("\n> Testing the model...")
    pred = model.predict(X_test)

    plt.imshow(X_test[30].reshape(96, 96), cmap='gray')
    plt.scatter(pred[30][0::2], pred[33][1::2], c='b', marker='.')
    plt.show()
