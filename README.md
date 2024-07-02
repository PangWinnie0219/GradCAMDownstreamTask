# Rethinking Feature Extraction: Gradient-based Localized Feature Extraction for End-to-End Surgical Downstream Tasks

This repository contains the reference code for the paper "[Rethinking Feature Extraction: Gradient-based Localized Feature Extraction for End-to-End Surgical Downstream Tasks](https://discovery.ucl.ac.uk/id/eprint/10159683/1/Rethinking_Feature_Extraction_Gradient-based_Localized_Feature_Extraction_for_End-to-End_Surgical_Downstream_Tasks.pdf)." To learn more about the project, check out our [presentation video](https://www.youtube.com/watch?v=wPvV8IvS2tE).

We develop a detector-free gradient-based localized feature extraction approach that enables end-to-end model training for downstream surgical tasks such as
report generation and tool-tissue interaction graph prediction. We eliminate the need for object detection or region proposal and feature extraction networks by extracting the features of interest from the discriminative regions in the feature map of the classification models. Here, the discriminative regions are
localized using gradient-based localization techniques (e.g. Grad-CAM). We show that our proposed approaches enable the realtime deployment of end-to-end models for surgical downstream tasks.

<p align="center">
  <img src="https://github.com/PangWinnie0219/GradCAMDownstreamTask/blob/master/figures/Graphical_abstract.jpg" alt="architecture" width="80%" height="80%">
</p>

If you find our code or paper useful, please cite as
```
@article{pang2022rethinking,
  title={Rethinking Feature Extraction: Gradient-Based Localized Feature Extraction for End-To-End Surgical Downstream Tasks},
  author={Pang, Winnie and Islam, Mobarakol and Mitheran, Sai and Seenivasan, Lalithkumar and Xu, Mengya and Ren, Hongliang},
  journal={IEEE Robotics and Automation Letters},
  volume={7},
  number={4},
  pages={12623--12630},
  year={2022},
  publisher={IEEE}
}
```

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

We are using the dataset from [Cholec80](http://camma.u-strasbg.fr/datasets) and [Robotic Instrument Segmentation Dataset from MICCAI2018 Endoscopic Vision Challenge](https://endovissub2018-roboticscenesegmentation.grand-challenge.org/Data/).

Cholec80 dataset: As the tissue label is required for captioning and interaction tasks, we added one extra label at the end of the original tool annotations of **all samples**, as shown in figure below. Since many types of tissues are present in the Cholec80 datasets (e.g. gallbladder, cystic plate and liver), the tissue label added in this work does not refer to the specific tissue but referring to the interacting tissue. For simplicity, we assume interacting tissue appears at all the frames in Cholec80 dataset.

![cholec80](https://github.com/PangWinnie0219/GradCAMDownstreamTask/blob/master/figures/cholec80_tool_label.jpg)


## Classification Model Training
Run `get_paths_labels.py` to generate the files needed for the training. Then, run `baseline.py` to start training the classification model. Ensure `save` is set to `True` as this checkpoint will be used for visualization and feature extraction later. 

Otherwise, you can downloaded the trained model file: [`GC-A`](https://drive.google.com/file/d/1m2gJejiBO1Z-SEFG2GOmFZAIkBzabdLv/view?usp=drive_link), [`GC-B`](https://drive.google.com/file/d/1jvAN2XKf8Lut-Qs69iZb463wua6Nglzg/view?usp=drive_link), [`GC-C`](https://drive.google.com/file/d/1IwIBuE5SEScyYQ4nGX_PG9XATDfVTDYN/view?usp=drive_link), [`GC-D`](https://drive.google.com/file/d/1VgrOe5pWBH2D53_Al72EjDV4ukcfsMVj/view?usp=drive_link). 

Place the trained model file inside the `./best_model_checkpoints`.

The result reported in the main table is from [`GC-D`](https://drive.google.com/file/d/1VgrOe5pWBH2D53_Al72EjDV4ukcfsMVj/view?usp=drive_link).

| Model  | Cholec80 | Endovis18 | Class Label |
| ----- | ------------- | ------------- | ------------- |
| GC-A  |       | Training  | bipolar forceps, prograsp forceps, large needle driver, clip applier, monopolar curved scissors, suction, ultrasound probe, stapler, tissue  |
| GC-B  | Training  | Fine-tuning  | bipolar forceps, prograsp forceps, large needle driver, clip applier, monopolar curved scissors, suction, ultrasound probe, stapler, tissue  |
| GC-C  | Training  | Fine-tuning  | bipolar forceps, prograsp forceps, large needle driver, clip applier, monopolar curved scissors, suction, ultrasound probe, stapler, **hook**, **specimen bag**, tissue  |
| GC-D  | Training  | Training |bipolar forceps, prograsp forceps, large needle driver, clip applier, monopolar curved scissors, suction, ultrasound probe, stapler, **hook**, **specimen bag**, tissue |

*hook* and *specimen bag* only appear in Cholec80 dataset.

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


**Examples of the Grad-CAM heatmap and bounding box can be found in the [supplementary video](https://drive.google.com/file/d/1aBYgbjTu8fPJPZBepve7lg24Uy2L3qSw/view?usp=sharing).**

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
 
## Contact
If you have any questions or feedback about this project, feel free to contact me at [winnie_pang@u.nus.edu](mailto:winnie_pang@u.nus.edu).
