"""
Note            : get rectangle coordinates and show boundary box based on gradcam output

GradCAM code modified from https://github.com/jacobgil/pytorch-grad-cam
bbox code modified from https://github.com/zalkikar/BBOX_GradCAM/blob/master/BBOXES_from_GRADCAM.py
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import os
from torchvision.utils import save_image


"""----------------------------------------------------------------------------------------------------
                                        Form Bounding Box
-------------------------------------------------------------------------------------------------------

---------------- Colour of bounding box ------------------------
bipolar_forceps             (class 0) => red: (0,0,255)
prograsp_forceps            (class 1) => blue: (255,0,0)
large_needle_driver         (class 2) => green: (0,255,0)
monopolar_curved_scissors   (class 3) => white: (255,255,255)
ultrasound_probe            (class 4) => yellow: (0,255,255)
suction                     (class 5) => purple: (255,0,255)
clip_applier                (class 6) => light blue: (255,255,0)
stapler                     (class 7) => dark green: (116, 139, 69)
hook                        (class 8) => brick: (31, 102, 156)
specimen bag                (class 9) => grey: (128, 128, 128)
tissue                      (class 10) => orange: (3, 97, 255)
"""

class_colour = [
    (0, 0, 255),
    (255, 0, 0),
    (0, 255, 0),
    (255, 255, 255),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (116, 139, 69),
    (31, 102, 156),
    (147, 20, 255),
    (3, 97, 255),
]


def form_bboxes(grey_img, rgb_img, threshold, colour):
    """apply threshold to the grey image, form and plot bbox using OpenCV
    input: grey image (0~1), RGB image, bbox_threshold, colour of bboc
    output: RGB image with bbox, list contains the bbox coordinates of the class, confidence of the bbox (averaging the grey value)
    """
    # grey_img is between 0 ~ 1
    coordinate_list = []
    ave_conf_list = []
    ret, thresh = cv2.threshold(grey_img, threshold, 255, cv2.THRESH_BINARY)
    thresh = thresh.astype(np.uint8)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(
            cnt
        )  # x, y is the top left corner, and w, h are the width and height respectively
        bbox_size = w * h
        if bbox_size > 25:
            coordinate_list.append([x, y, x + w, y + h])
            cv2.rectangle(rgb_img, (x, y), (x + w, y + h), colour, 2)
            cam_for_conf = grey_img[y : y + h, x : x + w]
            ave_conf = np.mean(cam_for_conf)
            ave_conf_list.append(ave_conf)

    cv2.waitKey()
    return rgb_img, coordinate_list, ave_conf_list


def plot_multiplebbox(
    args, cam, input_tensor, pred_cls, rgb_img, threshold, gt_bbox_list=None
):
    """plot and save bbox for all present classes in the image
    input: args, CAM object, input image, prediction output of the model (0 or 1), RGB image, bbox_threshold, GT bbox coordinates
    """
    nor_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    nor_img = 255 * ((nor_img - np.min(nor_img)) / (np.max(nor_img) - np.min(nor_img)))
    nor_img = nor_img.astype(np.uint8)

    save_image_name = "bbox_" + str(threshold) + ".png"
    coordinate_list_all_cls = []

    for cls in range(args.cls):
        if pred_cls[cls]:
            temp_list = [cls]
            grayscale_cam = cam(
                input_tensor=input_tensor, target_category=cls
            )  # grayscale_cam: (bs,H,W)
            grayscale_cam = grayscale_cam[0, :]

            bbox, coordinate_list, ave_conf_list = form_bboxes(
                grayscale_cam, nor_img, threshold, class_colour[cls]
            )

            ## combine all bbox in tissue class and save as one bbox
            # if cls != args.cls-1:
            #     bbox, coordinate_list, ave_conf_list = form_bboxes(grayscale_cam, nor_img, threshold, class_colour[cls])
            # elif cls == args.cls-1:
            #     threshold_cam = np.copy(grayscale_cam)
            #     threshold_cam[threshold_cam<threshold] = 0
            #     x1, y1, x2, y2 = bbox_coor(threshold_cam)
            #     cv2.rectangle(nor_img, (x1,y1), (x2,y2), class_colour[10], 2)
            #     coordinate_list = [x1, y1, x2, y2]

            temp_list.append(coordinate_list)
            coordinate_list_all_cls.append(temp_list)

        else:
            temp_list = [cls]
            temp_list.append([])
            coordinate_list_all_cls.append(temp_list)

    cv2.imwrite(save_image_name, bbox)
    cv2.destroyAllWindows()
    return coordinate_list_all_cls


def plot_gtbox(args, rgb_img, gt_bbox_list):
    """plot and save the GT bbox on RGB image
    input: args, RGB image, GT bbox list
    """
    nor_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    nor_img = 255 * ((nor_img - np.min(nor_img)) / (np.max(nor_img) - np.min(nor_img)))
    nor_img = nor_img.astype(np.uint8)

    save_image_name = "bbox_gt.png"
    for cls in range(args.cls):
        if cls < 8:
            x1, y1, x2, y2 = normalize_gtbbox(args, cls, gt_bbox_list)
            cv2.rectangle(nor_img, (x1, y1), (x2, y2), class_colour[cls], 2)

        if cls == args.cls - 1:
            x1, y1, x2, y2 = normalize_gtbbox(args, 8, gt_bbox_list)
            cv2.rectangle(nor_img, (x1, y1), (x2, y2), class_colour[10], 2)

    cv2.imwrite(save_image_name, nor_img)
    cv2.destroyAllWindows()


"""----------------------------------------------------------------------------------------------------
                                        Crop Bounding Box
-------------------------------------------------------------------------------------------------------
"""


def crop_bbox(coordinate, rgb_img):
    """crop the images according to the bbox coordinates
    input: bbox coordinates ([x1,y1,x2,y2]), RGB image
    output: cropped image
    """
    x1, y1, x2, y2 = coordinate[0], coordinate[1], coordinate[2], coordinate[3]
    crop_region = rgb_img[y1:y2, x1:x2]
    return crop_region


def crop_multibbox(args, rgb_img, coordinate_list_all_cls, frame_path):
    """crop and save all RGB images based on the bbox coordinates
    input: args, RGB image, list containing all bbox coordinates, path to save the cropped image
    """
    nor_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    nor_img = 255 * ((nor_img - np.min(nor_img)) / (np.max(nor_img) - np.min(nor_img)))
    nor_img = nor_img.astype(np.uint8)

    for cls in range(len(coordinate_list_all_cls) - 1):
        coordinate_list = coordinate_list_all_cls[cls][1]
        for i in range(len(coordinate_list)):
            x1 = coordinate_list[i][0]
            y1 = coordinate_list[i][1]
            x2 = coordinate_list[i][2]
            y2 = coordinate_list[i][3]
            crop_region = crop_bbox(coordinate_list[i], nor_img)
            bbox_name = frame_path + "_{}_{}_{},{},{},{}.png".format(
                cls, i, x1, y1, x2, y2
            )
            cv2.imwrite(bbox_name, crop_region)

    # for tissue class
    coordinate_list = coordinate_list_all_cls[args.cls - 1][1]
    if coordinate_list:
        x1 = min(x[0] for x in coordinate_list)
        y1 = min(x[1] for x in coordinate_list)
        x2 = max(x[2] for x in coordinate_list)
        y2 = max(x[3] for x in coordinate_list)
        combined_bbox = [x1, y1, x2, y2]
        crop_region = crop_bbox(combined_bbox, nor_img)
        bbox_name = frame_path + "_{}_{}_{},{},{},{}.png".format(10, 0, x1, y1, x2, y2)
        cv2.imwrite(bbox_name, crop_region)


"""----------------------------------------------------------------------------------------------------
                                        mIoU Calculation
-------------------------------------------------------------------------------------------------------
"""


def normalize_gtbbox(args, cls, gt_bbox_list):
    """scale the size of GT bbox to the size of image
    input: args, class number, GT bbox list
    output: scaled GT bbox coordintes
    """
    x1 = int(gt_bbox_list[cls * 4])
    y1 = int(gt_bbox_list[(cls * 4) + 1])
    x2 = int(gt_bbox_list[(cls * 4) + 2])
    y2 = int(gt_bbox_list[(cls * 4) + 3])

    x1 = x1 * args.imgw / 1280
    x2 = x2 * args.imgw / 1280
    y1 = y1 * args.imgh / 1024
    y2 = y2 * args.imgh / 1024

    return int(x1), int(y1), int(x2), int(y2)


def compute_iou(boxA, boxB):
    """ref: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    compute the IoU of 2 given bbox coordinates
    input: bbox coordinates (x1, y1, x2, y2)
    output: IoU
    """

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area

    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def compute_multiiou(args, gt_bbox_list, coordinate_list_all_cls):
    """compute the IoU for every class in frame-wise
    4 conditions:   1. both GT_bbox and pred_bbox appear: compute IoU using compute_iou
                    2. pred_bbox appear only: IoU = 0
                    3. GT_bbox appear only: IoU = 0
                    4. both bbox does not appear: IoU = NaN (does not contribute to mIoU)
    input: args, GT bbox list, predicted bbox list
    output: list of IoU for every class [1, num_class]
    """

    iou_list_all = []

    """------------------------ for instruments classes -----------------------------------"""
    # for 9 classes in miccai18 dataset, handle only 8 instrument classes
    for cls in range(8):
        gt_bbox = normalize_gtbbox(args, cls, gt_bbox_list)
        coordinate_list = coordinate_list_all_cls[cls][1]
        iou_list = []

        # case 1 and case 2
        if coordinate_list:
            for i in range(
                len(coordinate_list)
            ):  # all pred_bbox appear in the same class
                x1, y1, x2, y2 = coordinate_list[i]
                pred_xmin = x1
                pred_ymin = y1
                pred_xmax = x2
                pred_ymax = y2
                pred_bbox = [pred_xmin, pred_ymin, pred_xmax, pred_ymax]
                iou = compute_iou(gt_bbox, pred_bbox)
                iou_list.append(iou)
            iou_list_all.append(np.nanmean(iou_list))

        # case 3
        elif gt_bbox != (0, 0, 0, 0):
            print("####", gt_bbox)
            iou_list_all.append(0)

        # case 4
        else:
            iou_list_all.append(np.nan)

    """----------------------------- for tissue class ---------------------------------------"""
    ## iou computation for tissue, combine all tissue bbox as one bbox
    gt_bbox = normalize_gtbbox(args, 8, gt_bbox_list)  # gt_bbox_list has only 9 classes
    cls = args.cls - 1  # assume tissue class is at the last
    coordinate_list = coordinate_list_all_cls[cls][1]

    # case 1 and case 2
    if coordinate_list:
        x1 = min(x[0] for x in coordinate_list)
        y1 = min(x[1] for x in coordinate_list)
        x2 = max(x[2] for x in coordinate_list)
        y2 = max(x[3] for x in coordinate_list)
        pred_bbox = [x1, y1, x2, y2]

    # case 3
    else:
        pred_bbox = [0, 0, 0, 0]

        # case 4
        if not gt_bbox:
            iou_list_all.append(np.nan)

    iou = compute_iou(gt_bbox, pred_bbox)
    iou_list_all.append(iou)

    print(iou_list_all)
    return iou_list_all


"""----------------------------------------------------------------------------------------------------
                                     mAP Calculation for Bounding Box
-------------------------------------------------------------------------------------------------------
"""
# source: https://github.com/ultralytics/yolov5/blob/9cf80b7f6098391f70efa01b4fd156143752ef47/utils/metrics.py


def plot_multiplebbox_conf(
    args, cam, input_tensor, pred_cls, rgb_img, threshold, gt_bbox=None
):
    """plot multiple bbox for all present classes in the image
    input: args, CAM object, input image, predicted class from model [1,num_class], RGB image, bbox_threshold, GT bbox list
    output: list for bbox coordinate and list for confidence (for mAP calculation)
    """
    nor_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    nor_img = 255 * ((nor_img - np.min(nor_img)) / (np.max(nor_img) - np.min(nor_img)))
    nor_img = nor_img.astype(np.uint8)

    coordinate_list_all_cls = []
    ave_conf_list_all_cls = []
    gt_bbox_list = []
    pred_bbox_list = []

    for cls in range(args.cls):
        if pred_cls[cls]:
            temp_list = [cls]
            grayscale_cam = cam(
                input_tensor=input_tensor, target_category=cls
            )  # grayscale_cam: (bs,H,W)
            grayscale_cam = grayscale_cam[args.fidx, :]

            bbox, coordinate_list, ave_conf_list = form_bboxes(
                grayscale_cam, nor_img, threshold, class_colour[cls]
            )

            temp_list.append(coordinate_list)
            coordinate_list_all_cls.append(temp_list)

            if len(ave_conf_list) != 0:
                pred_bbox_list.extend([cls] * len(ave_conf_list))
                ave_conf_list_all_cls.extend(ave_conf_list)

            if cls < 8:
                gt_bbox_nor = normalize_gtbbox(args, cls, gt_bbox)
                if gt_bbox_nor != (0, 0, 0, 0):
                    gt_bbox_list.append(cls)
            elif cls == args.cls - 1:
                gt_bbox_nor = normalize_gtbbox(
                    args, 8, gt_bbox
                )  # GT bbox only has 9 class, tissue class at the last
                if gt_bbox_nor != (0, 0, 0, 0):
                    gt_bbox_list.append(cls)
        else:
            if cls < 8:
                gt_bbox_nor = normalize_gtbbox(args, cls, gt_bbox)
                if gt_bbox_nor != (0, 0, 0, 0):
                    gt_bbox_list.append(cls)

    cv2.destroyAllWindows()
    return coordinate_list_all_cls, gt_bbox_list, pred_bbox_list, ave_conf_list_all_cls


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, names=(), eps=1e-16):
    """compute the average precision, given the recall and precision curves.
    ref: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    input:
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    output: average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + eps)  # recall curve
            r[ci] = np.interp(
                -px, -conf[i], recall[:, 0], left=0
            )  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)

    i = f1.mean(0).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype("int32")


def compute_ap(recall, precision):
    """compute the average precision, given the recall and precision curves
    input: recall curve (list), precision curve (list)
    output: average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = "interp"  # methods: 'continuous', 'interp'
    if method == "interp":
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


"""----------------------------------------------------------------------------------------------------
                                Extract Features from Bounding Box
-------------------------------------------------------------------------------------------------------
"""


def scale_bbox(args, coordinates, feature_map_size):
    """scale the size of the bbox to the size of feature map
    input: args, bbox coordinate, size of feature map
    output: scaled bbox (x1, y1, x2, y2)
    """
    x1 = coordinates[0]
    y1 = coordinates[1]
    x2 = coordinates[2]
    y2 = coordinates[3]

    feature_map_x = feature_map_size[0]
    feature_map_y = feature_map_size[1]

    x1 = x1 * feature_map_x / args.imgw
    y1 = y1 * feature_map_y / args.imgh
    x2 = x2 * feature_map_x / args.imgw
    y2 = y2 * feature_map_y / args.imgh

    x1 = int(round(x1))
    y1 = int(round(y1))
    x2 = int(round(x2))
    y2 = int(round(y2))

    if x2 - x1 == 0:
        if x2 != feature_map_x:
            x2 = x1 + 1
        else:
            x1 = x1 - 1

    if y2 - y1 == 0:
        if y2 != feature_map_y:
            y2 = y1 + 1
        else:
            y1 = y1 - 1

    return [int(x1), int(y1), int(x2), int(y2)]


def extract_feature(args, coordinate, feature_outputs, threshold, pool_size):
    """extract features of a bbox
    input: args, predicted bbox coordinate, feature map layer [layer2, layer3, layer4], area_threshold, adaptive average pooling size
    output: extracted feature [512]
    """
    x1 = coordinate[0]
    y1 = coordinate[1]
    x2 = coordinate[2]
    y2 = coordinate[3]

    bbox_size = (x2 - x1) * (y2 - y1)
    if bbox_size < threshold[0]:  # [0]: smaller threshold
        feature_layer = feature_outputs[0]  # [0]: layer 2
    elif bbox_size < threshold[1]:
        feature_layer = feature_outputs[1]
    else:
        feature_layer = feature_outputs[2]

    feature_size = feature_layer.size()

    scaled_coor = scale_bbox(args, coordinate, [feature_size[3], feature_size[2]])
    bbox_feature = feature_layer[
        :, :, scaled_coor[1] : scaled_coor[3], scaled_coor[0] : scaled_coor[2]
    ]  # exp: torch.Size([1, 1024, 4, 6])

    att = F.adaptive_avg_pool2d(bbox_feature, [pool_size[1], pool_size[2]]).permute(
        0, 2, 3, 1
    )  # exp: torch.Size([1, 1, 1, 1024])

    att_size = att.size()
    if att_size[3] != pool_size[0]:
        att = F.adaptive_avg_pool2d(att, [pool_size[1], pool_size[0]]).permute(
            0, 1, 2, 3
        )  # exp: torch.Size([1, 1, 1, 512])

    att = torch.flatten(att)  # torch.Size([512])

    ## To check if have 'nan' value in the extracted features due to small bbox area (Sanity Check)
    feature_sum = np.sum(att.data.cpu().float().numpy())
    array_has_nan = np.isnan(feature_sum)
    if array_has_nan:
        print(
            "!!!!!!!!!!",
            (x2 - x1) * (y2 - y1),
            scaled_coor,
            (scaled_coor[2] - scaled_coor[0]) * (scaled_coor[3] - scaled_coor[1]),
            bbox_feature.shape,
        )  # bbox_feature = torch.Size([1, 512, 6, 0])
    return att


def extract_feature_all(
    args,
    frame_path_bbox,
    frame_path_frame,
    coordinate_list_all_cls,
    feature_outputs,
    threshold,
    pool_size,
):
    """extract and save features of all bbox in one image
    save features by (1) bbox and (2) frame
    input: args, path to save bbox features, path to save frame features, predicted bbox list,
            feature map [layer2, layer3, layer4], area_threshold, adaptive average pooling size
    """
    frame_features = []

    """----------------------------- for instrument classes ---------------------------------------"""
    for cls in range(len(coordinate_list_all_cls) - 1):
        coordinate_list = coordinate_list_all_cls[cls][1]
        for i in range(len(coordinate_list)):
            x1 = coordinate_list[i][0]
            y1 = coordinate_list[i][1]
            x2 = coordinate_list[i][2]
            y2 = coordinate_list[i][3]
            att = extract_feature(
                args, coordinate_list[i], feature_outputs, threshold, pool_size
            )
            bbox_name = frame_path_bbox + "_{}_{}_{},{},{},{}".format(
                cls, i, x1, y1, x2, y2
            )
            print(bbox_name)
            np.savez_compressed(bbox_name, feat=att.data.cpu().float().numpy())
            frame_features.append(att.data.cpu().float().numpy())

    """----------------------------- for tissue class -------------------------------------------"""
    coordinate_list = coordinate_list_all_cls[args.cls - 1][1]
    if coordinate_list:
        x1 = min(x[0] for x in coordinate_list)
        y1 = min(x[1] for x in coordinate_list)
        x2 = max(x[2] for x in coordinate_list)
        y2 = max(x[3] for x in coordinate_list)
        combined_bbox = [x1, y1, x2, y2]
        att = extract_feature(
            args, combined_bbox, feature_outputs, threshold, pool_size
        )
        bbox_name = frame_path_bbox + "_{}_{}_{},{},{},{}".format(10, 0, x1, y1, x2, y2)
        print(bbox_name)
        np.savez_compressed(bbox_name, feat=att.data.cpu().float().numpy())
        frame_features.append(att.data.cpu().float().numpy())

    ## append all bbox features together and save as frame features
    frame_features = np.array(frame_features)
    print(frame_path_frame, frame_features.shape)  # (num_bbox, feature_size)
    np.save(frame_path_frame, frame_features)


"""----------------------------------------------------------------------------------------------------
                                Extract Features from Bounding Box
-------------------------------------------------------------------------------------------------------
"""


def bbox_coor(heatmap):
    """get the coordinates of rectangle/bbox that cover the region of interest
    input: grayscale vale [0~1]
    output: bbox coordinate (x1, y1, x2, y2)
    """
    rows = np.any(heatmap, axis=1)
    cols = np.any(heatmap, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return xmin, ymin, xmax + 1, ymax + 1


def heatmap_extract_feature(
    args,
    cam,
    pred_cls,
    input_tensor,
    feature_outputs,
    threshold,
    feature_path_bbox,
    feature_path_frame,
    pool_size,
):
    """extract features from feature map based on gradcam heatmap
    save the features by (1) bbox and (2) frame
    input: args, CAM object, prediction from model [1, num_class], feature maps [layer2, layer3, layer4],
            bbox_threshold, path to save bbox features, path to save frame features, adaptive average pooling size
    """

    feature_map = feature_outputs[2]  # [1] => layer 3: [1,1024,16,20]
    feature_map_c = feature_map.shape[1]  # 1024
    feature_map_h = feature_map.shape[2]  # 16
    feature_map_w = feature_map.shape[3]  # 20

    frame_features = []
    for cls in range(args.cls):
        if pred_cls[cls]:
            heat_map = cam(
                input_tensor=input_tensor,
                target_category=cls,
                target_size=(feature_map_w, feature_map_h),
            )  # grayscale_cam: (bs,H,W)
            heat_map = torch.from_numpy(heat_map)  # [1,16,20]

            heat_map = heat_map.expand(
                feature_map_c, 1, feature_map_h, feature_map_w
            )  # torch.Size([1024, 1, 16, 20])

            # heat_map.shape = torch.Size([1024, 1, 16, 20])
            # heat_map[:,0].shape = torch.Size([1024, 16, 20])
            # feature_map.shape = torch.Size([1, 1024, 16, 20]

            ## ref: https://github.com/mobarakol/tutorial_notebooks/blob/main/heatmap_to_bbox.ipynb
            #### v1: feature map as obj feature
            feature_map_cls = feature_map[
                heat_map[None, :, 0] > threshold
            ]  # feature_map_cls[None,None] = torch.Size([1, 1, 94208]) <= example

            att = F.adaptive_avg_pool1d(
                feature_map_cls[None, None], [pool_size[0]]
            ).permute(
                2, 0, 1
            )  # torch.Size([128,1,1])
            att = F.adaptive_avg_pool2d(
                att[None], [pool_size[1], pool_size[2]]
            ).permute(
                0, 1, 2, 3
            )  # torch.Size([1, 128, 2, 2])
            att = torch.flatten(att)  # torch.Size([512])

            feature_name = feature_path_bbox + "_{}_{}_{},{},{},{}".format(
                cls, 0, 0, 0, args.imgw, args.imgh
            )
            np.savez_compressed(feature_name, feat=att.data.cpu().float().numpy())
            frame_features.append(att.data.cpu().float().numpy())

        else:
            continue

    frame_features = np.array(frame_features)
    print(feature_path_frame, frame_features.shape)  # (num_pred_cls, feature_size)
    np.save(feature_path_frame, frame_features)
