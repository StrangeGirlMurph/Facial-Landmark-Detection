import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

print("\n> Loading the data...")
# -- loading data --
Train_Dir = '../data/training/training.csv'
Test_Dir = '../data/test/test.csv'
lookid_dir = '../data/IdLookupTable.csv'
train_data = pd.read_csv(Train_Dir)
test_data = pd.read_csv(Test_Dir)
lookid_data = pd.read_csv(lookid_dir)
os.listdir('../data')

# print(train_data.head().T)

# -- preprocessing --
# - missing values -
print("- Checking for missing values...")
# print(train_data.isnull().sum())

# fill missing values with previous value
train_data.fillna(method='ffill', inplace=True)
# remove rows with missing values (would result in having only 2140 images to train on)
#train_data.reset_index(drop = True,inplace = True)

print("- Missing values after filling:")
print(train_data.isnull().any().value_counts())

# - prepare features -
images = []
for i in range(0, 7049):
    img = train_data['Image'][i].split(' ')
    img = ['0' if x == '' else x for x in img]
    images.append(img)

image_list = np.array(images, dtype='float')
X_train = image_list.reshape(-1, 96, 96, 1)

# - prepare labels -
training = train_data.drop('Image', axis=1)

y_train = []
for i in range(0, train_data.shape[0]):
    y = training.iloc[i, :]
    y_train.append(y)

y_train = np.array(y_train, dtype='float')

# example image with keypoints
plt.imshow(X_train[30].reshape(96, 96), cmap='gray')
plt.scatter(y_train[30][0::2], y_train[33][1::2], c='b', marker='.')
plt.show()

# - preparing test data -
timages = []
for i in range(0, 1783):
    timg = test_data['Image'][i].split(' ')
    timg = ['0' if x == '' else x for x in timg]

    timages.append(timg)

timage_list = np.array(timages, dtype='float')
X_test = timage_list.reshape(-1, 96, 96, 1)
