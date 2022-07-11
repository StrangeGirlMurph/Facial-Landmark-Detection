# Deep Learning - Facial Landmark Detection
A deep learning model to detect facial landmarks from images/videos.

_Disclaimer: Git LFS is used for this repository! The repo contains the dataset. I recommend cloning it in uni if you're using a laptop (check [Usage](#usage))._

+ [Overleaf Project](https://www.overleaf.com/8268422246bjnxsrvsbqxn) for the report
+ [Demo](https://colab.research.google.com/github/StrangeGirlMurph/DeepLearning-FacialLandmarkDetection/blob/master/demo.ipynb) on colab
+ [Dataset on Kaggle](https://www.kaggle.com/c/facial-keypoints-detection/overview)
+ [Organisational Slides](https://docs.google.com/presentation/d/1Lbggpj_nj4RomOm4q35XUcoOoDsIDvT18GLpOIygC2Q/edit#slide=id.p)
+ [GitHub Repo](https://github.com/StrangeGirlMurph/DeepLearning-FacialLandmarkDetection)

# Working-Area
## Questions
+ Do we really have to name the files according to the notes on the slides?

## Ideas
+ Simulate the rotation of the faces.

## Notes
+ The test dataset doesn't include labels for the data.
+ Changing the tensorflow log-level (powershell)
  + for that terminal: `$Env:TF_CPP_MIN_LOG_LEVEL = "3"`
  + permanently: `Add-Content -Path $Profile.CurrentUserAllHosts -Value '$Env:TF_CPP_MIN_LOG_LEVEL = "3"'`

# Usage
Install Git LFS with `git lfs install`. Clone/pull with `git lfs clone/pull`.  
To run execute the `main.py` file.
