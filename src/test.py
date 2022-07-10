# evaluation metric functions
import matplotlib.pyplot as plt
from data_loading import X_test
from architecture import model
import train

print("\n> Testing the model...")
pred = model.predict(X_test)

plt.imshow(X_test[30].reshape(96, 96), cmap='gray')
plt.scatter(pred[30][0::2], pred[33][1::2], c='b', marker='.')
plt.show()
