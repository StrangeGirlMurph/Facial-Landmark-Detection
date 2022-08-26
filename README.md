# Deep Learning - Facial Landmark Detection
A deep learning model to detect facial landmarks from images/videos.

_Disclaimer: Git LFS is used for this repository! The repo contains the dataset. Check [Usage](#usage) how to clone the repo and pull._

+ [Overleaf Project](https://www.overleaf.com/8268422246bjnxsrvsbqxn) for the report
+ [Dataset on Kaggle](https://www.kaggle.com/c/facial-keypoints-detection/overview)
+ [Organisational Slides](https://docs.google.com/presentation/d/1Lbggpj_nj4RomOm4q35XUcoOoDsIDvT18GLpOIygC2Q/edit#slide=id.p)
+ [GitHub Repo](https://github.com/StrangeGirlMurph/DeepLearning-FacialLandmarkDetection)

# Usage
Install Git LFS with `git lfs install` and clone/pull the large files with `git lfs clone/pull`.  
To train the model set the parameters and execute `mainTraining.py`. To test a model do the same with `mainTesting.py`

# Working-Area
## Questions
+ Demo file? Complete code from raw data to video feed?

## Ideas
+ Applying augmentations to the data (rotation, brightness, horizontal/vertical shift, random noise).
+ Different typ of models to compare to (heatmap approach).
+ Annotating a video to get technical measurements.
+ Flipping
+ more angles

## Things to deal with
+ The video window doesn't have fixed aspect ratio... (opencv sucks...)

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
  + Specialties ⁘ more augmentation (rotation, horizontal flip, crop & pad, perspective, brightness & contrast) + more epochs
  
## Notes
+ look for labeled video
+ Murph
  + Dataset
  + Masking
  + demo
  + bug
---
+ The "test dataset" doesn't include labels for the data.
+ Please don't train on a CPU. Don't be as dumb as I am. 🥲

## Resources
+ [Fast Facial Landmark Detection and Applications: A Survey](https://arxiv.org/pdf/2101.10808.pdf)
+ [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace)
+ [Transfer learning keras](https://keras.io/guides/transfer_learning/)

## Tips
+ Changing the tensorflow log-level (powershell)
  + for that terminal: `$Env:TF_CPP_MIN_LOG_LEVEL = "3"`
  + permanently: `Add-Content -Path $Profile.CurrentUserAllHosts -Value '$Env:TF_CPP_MIN_LOG_LEVEL = "3"'`


# Process notes
## Masking
+ Setting the abs of the missing values to 0 and taking the mean reducing everything to a scalar.
+ Discarding the abs of all the missing values and taking the mean reducing everything to a scalar.
+ Discarding the abs of all the missing value and taking the mean per data point. For that I used ragged tensors which let the time for a single epoch go from 9s to 30s (9s * 100epochs = 15min, 30s * 100epochs = 50min)
+ Optimising the crap out of it and getting back to normal tensors -> 9s per epoch and correct calculation.
+ Again a logical mistake but I hope it's calculating the correct thing now...
