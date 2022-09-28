# Rethinking Feature Extraction: Gradient-based Localized Feature Extraction for End-to-End Surgical Downstream Tasks

This repository contains the reference code for the paper "Rethinking Feature Extraction: Gradient-based Localized Feature Extraction for End-to-End Surgical Downstream Tasks"


## Introduction

**To be added**


## Environment setup
1. Clone the repository 

```bash
git clone https://github.com/PangWinnie0219/GradCAMDownstreamTask.git
```
2. Install the packages required using the `requirements.txt` file:

```bash
pip install -r requirements.txt 
```

**Note: Python 3.6 is required to run our code.**


## Data Preprocessing

We are using the dataset from [Cholec80](http://camma.u-strasbg.fr/datasets) and Robotic Instrument Segmentation Dataset from MICCAI2018 Endoscopic Vision Challenge.

**To be added**

## Classification Model Training

Run `python3.6 baseline.py` to start training the Grad-CAM model. Ensure `save` is set to `True` as this checkpoint will be used for visualization and feature extraction later.

Otherwise, you can downloaded the trained model file:

- GC-A: [miccai2018_9class_ResNet50_256,320_32_lr_0.001_dropout_0.2_best_checkpoint.pth.tar] (**To be added**)
- GC-B: [miccai2018_9class_cholecResNet50_256,320_32_lr_0.001_dropout_0.2_best_checkpoint.pth.tar] (**To be added**)
- GC-C: [miccai2018_11class_cholec_ResNet50_256,320_32_lr_0.001_best_checkpoint.pth.tar] (**To be added**)
- GC-D: [combine_miccai18_ResNet50_256,320_170_best_checkpoint.pth.tar] (**To be added**)

Place the trained model file inside the `./best_model_checkpoints`.


## Grad-CAM Heatmap and Bounding Box Visualization

cd into the `utils` directory

```bash
cd utils
```

You can visualise the Grad-CAM heatmap and bounding box using

```bash
python3.6 miccai_bbox.py
```

In order to select a specific frame and heatmap of specific class, you can define them with `bidx` and `tclass` respectively. For example if you want to view the heatmap 
for class 3 of the 15th image in the dataset, you can run the following: 
```bash
python3.6 miccai_bbox.py --bidx 15  --tclass 3
```

The threshold, T_ROI can be defined using `threshold` to see the effect of thresholding to the bounding box generation. 


### Examples of the Grad-CAM heatmap and bounding box
**To be added**

## Feature Extraction 
Set the result_filename in the code to accordingly if you are training the Grad-CAM model from scratch. If you are using our checkpoint, set `gc_model` to `1`, `2` `3` or `4`
to load the checkpoint from GC-A, GC-B, GC-C and GC-D respectively. If you are using `gc_model` = `1` or `2`, set `cls` to 9, else, set `cls` to 11.


### 1. Variant 1: Localization and Naive FE (LN-FE)

This method is similar to the conventional feature extraction method. 
The region images will be cropped from the raw image and these cropped region images will be forwarded to the feature extractor.

1. Crop the region images based on the predicted bounding box 
```bash
python3.6 utils/crop_bbox.py 
```
2. Forward the cropped region image to the model again
```bash
python3.6 image_extract_feature.py
```

### 2. Variant 2: Localization-aided FE (L-FE)
The features is extracted from the feature map of the classification model based on the bounding box coordinates.
```bash
python3.6 bbox_extract_feature.py
```

### 3. Variant 3: Single-pass Localization-aided FE (SL-FE)
The features is extracted from the feature map of the classification model in a single-pass based on the heatmap (no bounding box generation).
```bash
python3.6 heatmap_extract_feature.py
```


    
## Acknowledgement
Code adopted and modified from : [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)

### Downstream Task
The features extracted can be used for the downstream task such as:

- Captioning
    * Paper: [Meshed-Memory Transformer for Image Captioning ](https://arxiv.org/abs/1912.08226)
    * Official implementation [code](https://github.com/aimagelab/meshed-memory-transformer)
- Interaction
    * Paper: [CogTree: Cognition Tree Loss for Unbiased Scene Graph Generation](https://arxiv.org/abs/2009.07526)
    * Official implementation [code](https://github.com/CYVincent/Scene-Graph-Transformer-CogTree)
