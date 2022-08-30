# Deep Learning - Facial Landmark Detection
A deep learning model to detect facial landmarks from images/videos.
We use Keras/TensorFlow and this [Dataset on Kaggle](https://www.kaggle.com/c/facial-keypoints-detection/overview).

_Disclaimer: Git LFS is used for this repository! The repo contains the dataset itself.  
Check [Usage](#usage) on how to clone the repo and pull._
<details>
<summary>Example image</summary>
<p align="center">
<img src="https://user-images.githubusercontent.com/62220780/187404422-171031f3-aa08-4549-997e-db40dc642dda.png" width="600">
</p>
</details>


## Usage
To test the model just follow the [Demo on Colab](https://colab.research.google.com/github/StrangeGirlMurph/Facial-Landmark-Detection/blob/master/demo.ipynb).

Otherwise:  
Install Git LFS with `git lfs install` and clone/pull the large files with `git lfs clone/pull`.  
To train the model set the parameters and execute `mainTraining.py`. To test or evaluate a model do the same with `mainTesting.py`/`mainEvaluating.py`

__Changing the tensorflow log-level (powershell):__
+ For that terminal instance: `$Env:TF_CPP_MIN_LOG_LEVEL = "3"`
+ Permanently: `Add-Content -Path $Profile.CurrentUserAllHosts -Value '$Env:TF_CPP_MIN_LOG_LEVEL = "3"'`

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

## Dataset references
### [Facial Keypoints Dataset](https://www.kaggle.com/c/facial-keypoints-detection/overview)
The dataset for the kaggle competition from where we have the image dataset was graciously provided by Dr. Yoshua Bengio of the University of Montreal.

### [300VW Dataset](https://ibug.doc.ic.ac.uk/resources/300-VW/)
To test on labeled video data we are using the 300VW dataset.

[1] J.Shen, S.Zafeiriou, G. S. Chrysos, J.Kossaifi, G.Tzimiropoulos, and M. Pantic. The first facial landmark tracking in-the-wild challenge: Benchmark and results. In IEEE International Conference on Computer Vision Workshops (ICCVW), 2015. IEEE, 2015.

[2] G. S. Chrysos, E. Antonakos, S. Zafeiriou and P. Snape. Offline deformable face tracking in arbitrary videos. In IEEE International Conference on Computer Vision Workshops (ICCVW), 2015. IEEE, 2015,

[3] G. Tzimiropoulos. Project-out cascaded regression with an application to face alignment. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 3659–3667, 2015. 
