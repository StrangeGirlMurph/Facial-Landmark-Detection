import matplotlib.pyplot as plt


def showImage(x, y, cmap="gray"):
    plt.imshow(x.reshape(96, 96), cmap=cmap)
    # all the even/odd entries are the x/y coordinates
    plt.scatter(y[0::2], y[1::2], c='b', marker='.')
    plt.show()
