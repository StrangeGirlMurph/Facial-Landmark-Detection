# Deep Learning - Facial Landmark Detection
A deep learning model to detect facial landmarks from images/videos. We use Keras/TensorFlow and this [Dataset on Kaggle](https://www.kaggle.com/c/facial-keypoints-detection/overview).

_Disclaimer: Git LFS is used for this repository! The repo contains the dataset. Check [Usage](#usage) how to clone the repo and pull._

## Usage
Install Git LFS with `git lfs install` and clone/pull the large files with `git lfs clone/pull`.  
To train the model set the parameters and execute `mainTraining.py`. To test a model do the same with `mainTesting.py`

__Changing the tensorflow log-level (powershell):__
+ for that terminal: `$Env:TF_CPP_MIN_LOG_LEVEL = "3"`
+ permanently: `Add-Content -Path $Profile.CurrentUserAllHosts -Value '$Env:TF_CPP_MIN_LOG_LEVEL = "3"'`

## Models
+ V1 was trained in 43 min on a Ryzen 5 3600. (Epochs: 20, Batch size: 256, Validation split: 0.2, #Images: 7049)
+ V2 was trained in 15 min on a Colab GPU. (Epochs: 100, Batch size: 256, Validation split: 0.2, #Images: 7049)
  + Results ⁘ loss: 4.3127 - masked_mean_absolute_error: 1.5840 - masked_accuracy: 0.5411
  + Specialties ⁘ masking the ouput for the missing values
+ V3 was trained in 25 min on a Colab GPU. (Epochs: 100, Batch size: 256, Validation split: 0.2, #Images: 11329)
  + Results ⁘ loss: 3.7041 - masked_mean_absolute_error: 1.4212 - masked_accuracy: 0.6283
  + Specialties ⁘ trained with rotation augmented data
+ V4 was trained in 40 min on a Colab GPU. (Epochs: 100, Batch size: 256, Validation split: 0.2, #Images: 17749)
  + Results ⁘ loss: 4.2459 - masked_mean_absolute_error: 1.5622 - masked_accuracy: 0.6248
  + Specialties ⁘ more augmentation (rotation, horizontal flip, crop & pad, perspective, brightness & contrast)
