import albumentations as A
import numpy as np
import cv2 as cv


def horizontalFlip(X, y, p=1):
    """Performs horizontal flip."""
    transform = A.Compose(
        [A.HorizontalFlip(p=p)],
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
    )
    transformed = transform(image=X, keypoints=y)
    return transformed['image'], transformed['keypoints']


def rotate(X, y, limit=45, p=1):
    """Performs a rotation augmentation."""
    transform = A.Compose(
        [A.Rotate(limit=limit, p=p, border_mode=cv.BORDER_REPLICATE)],
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
    )
    transformed = transform(image=X, keypoints=y)
    return transformed['image'], transformed['keypoints']


def cropAndPad(X, y, percent=(-0.05, 0.1), p=1):
    """Performs a cropping and padding augmentation."""
    transform = A.Compose(
        [A.CropAndPad(p=p, percent=percent, pad_mode=cv.BORDER_REPLICATE, sample_independently=False)],
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
    )
    transformed = transform(image=X, keypoints=y)
    return transformed['image'], transformed['keypoints']


def perspective(X, y, scale=(0.05, 0.05), p=1):
    """Performs a perspective transformation."""
    transform = A.Compose(
        [A.Perspective(p=p, scale=scale, pad_mode=cv.BORDER_REPLICATE)],
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
    )
    transformed = transform(image=X, keypoints=y)
    return transformed['image'], transformed['keypoints']


def brightnessAndContrast(X, y, brightness_limit=0.4, contrast_limit=0.4, p=1):
    """Performs a brightness and contrast augmentation."""
    transform = A.Compose(
        [A.RandomBrightnessContrast(brightness_limit=brightness_limit, contrast_limit=contrast_limit, brightness_by_max=False, p=p)],
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
    )
    transformed = transform(image=X.astype(np.uint8), keypoints=y)
    return transformed['image'].astype(np.float32), transformed['keypoints']


def performRotationAugmentation(X, y, angles=[12]):
    """Performs rotation augmentation on the given data."""
    X_new = np.empty(shape=(0, 96, 96), dtype=np.uint8)
    y_new = np.empty(shape=(0, 30), dtype=np.float16)

    # bring the labels in coordinate shape (x, y, 1)
    c = np.array([np.stack((yi[0::2], yi[1::2]), axis=1) for yi in y])
    c = np.pad(c, ((0, 0), (0, 0), (0, 1)), 'constant', constant_values=(1))

    for angle in angles:
        M = cv.getRotationMatrix2D((48, 48), angle, 1.0)
        X_new = np.concatenate((X_new, [cv.warpAffine(x, M, (96, 96), flags=cv.INTER_CUBIC) for x in X]))
        y_new = np.concatenate((y_new, [np.array([np.matmul(M, yi) for yi in ci], dtype=np.float16).flatten() for ci in c]))

    return X_new.reshape(-1, 96, 96, 1), y_new
