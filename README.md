# End-to-End Model Deployment Using Grad-CAM for Medical Downstream Task

This repository contains the reference code for the paper "End-to-End Model Deployment Using Grad-CAM for Medical Downstream Task"


## Introduction

To be added


## Environment setup
1. Clone the repository 

```
git clone https://github.com/PangWinnie0219/Grad-CAM.git
```
3. Install the packages required using the `requirements.txt` file:

```
pip install -r requirements.txt 
```

Note: Python 3.6 is required to run our code.


## Data Preprocessing

We are using the dataset from [Cholec80](http://camma.u-strasbg.fr/datasets) and Robotic Instrument Segmentation Dataset from MICCAI2018 Endoscopic Vision Challenge.

To be added

## Grad-CAM model training

Run `python3.6 baseline.py` to start training the Grad-CAM model. Ensure `save` is set to `True` as this checkpoint will be used for visualization and feature extraction later.

Otherwise, you can downloaded the trained model file:

- GC-A: [miccai2018_9class_ResNet50_256,320_32_lr_0.001_dropout_0.2_best_checkpoint.pth.tar](to be added)
- GC-B: [miccai2018_9class_cholecResNet50_256,320_32_lr_0.001_dropout_0.2_best_checkpoint.pth.tar] (to be added)
- GC-C: [miccai2018_11class_cholec_ResNet50_256,320_32_lr_0.001_best_checkpoint.pth.tar] (to be added)
- GC-D: [combine_miccai18_ResNet50_256,320_170_best_checkpoint.pth.tar] (to be added)

Place the trained model file inside the `./best_model_checkpoints`.


## Grad-CAM Heatmap and Bounding Box Visualization

You can visualise the Grad-CAM heatmap and bounding box using

```
python3.6 miccai_bbox.py
```

In order to select a specific frame and heatmap of specific class, you can define them with `bidx` and `tclass` respectively. For example if you want to view the heatmap 
for class 3 of the 15th image in the dataset, you can run the following: 
```
python3.6 miccai_bbox.py --bidx 15  --tclass 3
```

The threshold, T can be defined using `threshold` to see the effect of thresholding to the bounding box generation. 


### Examples of the Grad-CAM heatmap and bounding box


## Feature Extraction 
Set the result_filename in the code to accordingly if you are training the Grad-CAM model from scratch. If you are using our checkpoint, set `gc_model` to `1`, `2` `3` or `4`
to load the checkpoint from GC-A, GC-B, GC-C and GC-D respectively.


### 1. Method 1 (Raw Image):

This method is similar to the conventional feature extraction method. 
The region images will be cropped from the raw image and these cropped region images will be forwarded to the model again.

1. Crop the region images based on the predicted bounding box 
```
python3.6 crop_bbox.py 
```
2. Forward the cropped region image to the model again
```
python3.6 image_extract_feature.py
```

### 2. Method 2 (Localization + Detection):
The features is extracted from the feature map of the Grad-CAM model based on the bounding box coordinates.
```
python3.6 bbox_extract_feature.py
```

### 3. Method 3 (Localization):
The features is extracted from the feature map of the Grad-CAM model directly based on the heatmap (no bounding box generation).
```
python3.6 heatmap_extract_feature.py
```


    
## Acknowledgement
Code adopted and modified from : [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)

### Downstream Task
The features extracted can be used for the downstream task such as:

- Captioning
    * Paper: [Class-Incremental Domain Adaptation with Smoothing and Calibration for Surgical Report Generation](https://arxiv.org/pdf/2107.11091.pdf)
    * Official implementation [code](https://github.com/XuMengyaAmy/CIDACaptioning/blob/master/README.md)
- Interaction
    * Paper: [Global-Reasoned Multi-Task Learning Model for Surgical Scene Understanding](https://arxiv.org/pdf/2201.11957.pdf)
    * Official implementation [code](https://github.com/lalithjets/Global-reasoned-multi-task-model/blob/master/README.md)



















