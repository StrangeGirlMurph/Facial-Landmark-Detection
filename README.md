# Deep Learning - Facial Landmark Detection
A deep learning model to detect facial landmarks from images/videos.

_Disclaimer: Git LFS is used for this repository! The repo contains the dataset. Check [Usage](#usage) how to clone the repo and pull._

+ [Overleaf Project](https://www.overleaf.com/8268422246bjnxsrvsbqxn) for the report
+ [Demo](https://colab.research.google.com/github/StrangeGirlMurph/DeepLearning-FacialLandmarkDetection/blob/master/demo.ipynb) on colab
+ [Dataset on Kaggle](https://www.kaggle.com/c/facial-keypoints-detection/overview)
+ [Organisational Slides](https://docs.google.com/presentation/d/1Lbggpj_nj4RomOm4q35XUcoOoDsIDvT18GLpOIygC2Q/edit#slide=id.p)
+ [GitHub Repo](https://github.com/StrangeGirlMurph/DeepLearning-FacialLandmarkDetection)

# Working-Area
## Questions
+ Do we really have to name the files according to the names mentioned in the slides?
+ Problem: The model works quite well on the dataset but not on webcam input.
  + The dataset really isn't the best...
     + 68% of the data is missing some key points
     + The perspective is not really useful for input from a webcam or something
  + Should we try some form of augmentation? I saw people do all kinds of things
    + rotation, brightness, horizontal/vertical shift, random noise

## Ideas
+ Applying augmentations to the data (rotation, brightness, horizontal/vertical shift, random noise)

## Resources
+ [Fast Facial Landmark Detection and Applications: A Survey](https://arxiv.org/pdf/2101.10808.pdf)

## Notes
+ The "test dataset" doesn't include labels for the data.

## Tips
+ Changing the tensorflow log-level (powershell)
  + for that terminal: `$Env:TF_CPP_MIN_LOG_LEVEL = "3"`
  + permanently: `Add-Content -Path $Profile.CurrentUserAllHosts -Value '$Env:TF_CPP_MIN_LOG_LEVEL = "3"'`

## Models
+ V1 was trained in 43min on a Ryzen 5 3600. (Epochs: 20, Batch size: 256, Validation split: 0.2)

# Usage
Install Git LFS with `git lfs install` and clone/pull with `git lfs clone/pull`.  
To run execute the `main.py` file.
