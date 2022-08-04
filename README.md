# Deep Learning - Facial Landmark Detection
A deep learning model to detect facial landmarks from images/videos.

_Disclaimer: Git LFS is used for this repository! The repo contains the dataset. Check [Usage](#usage) how to clone the repo and pull._

+ [Overleaf Project](https://www.overleaf.com/8268422246bjnxsrvsbqxn) for the report
+ [Dataset on Kaggle](https://www.kaggle.com/c/facial-keypoints-detection/overview)
+ [Organisational Slides](https://docs.google.com/presentation/d/1Lbggpj_nj4RomOm4q35XUcoOoDsIDvT18GLpOIygC2Q/edit#slide=id.p)
+ [GitHub Repo](https://github.com/StrangeGirlMurph/DeepLearning-FacialLandmarkDetection)

# Working-Area
## Questions
+ "Proper way" to do masking instead of writing your own loss/metrics?
+ I am a bit scared that the optimizer doesn't train properly with the masking...

## Ideas
+ Applying augmentations to the data (rotation, brightness, horizontal/vertical shift, random noise).
+ Different typ of models to compare to (heatmap approach).
+ Annotating a video to get technical measurements.

## Things to deal with
+ The video window doesn't have fixed aspect ratio... (opencv sucks...)

## Models
+ V1 was trained in 43 min on a Ryzen 5 3600. (Epochs: 20, Batch size: 256, Validation split: 0.2)
+ V2 was trained in 15 min on a Colab GPU. (Epochs: 100, Batch size: 256, Validation split: 0.2)
  + loss: 8.4730 - masked_mean_absolute_error: 2.1146 - masked_accuracy: 0.5371

## Notes
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

# Usage
Install Git LFS with `git lfs install` and clone/pull with `git lfs clone/pull`.  
To run execute the `main.py` file.
