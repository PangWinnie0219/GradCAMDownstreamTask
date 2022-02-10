'''
Task: get rectangle coordinates and show boundary box based on gradcam output
BBOX code modified from https://github.com/zalkikar/BBOX_GradCAM/blob/master/BBOXES_from_GRADCAM.py

Functions in this file
1. plot_multiplebbox(args, cam, input_tensor, rgb_img, threshold, gt_bbox_list): 
    - plot bbox for all class
    - return the list with classes and bbox coordinates 
    coordinate_list_all_cls = [[0, [[bbox_ccor_1],[bbox_ccor_2]]],[1, [[bbox_coor]]],[2,[]],..,[10,[[bbox_coor]]]]

2. compute_multiiou (args, gt_bbox_list, coordinate_list_all_cls):
    - compute the IoU for every class in frame-wise

3. crop_multibbox(rgb_img, coordinate_list_all_cls, frame_path):
    - crop bbox and save it as an image

4. extract_feature_all(args, frame_path, coordinate_list_all_cls, feature_outputs, threshold):
    - extract and save features from the layers of the model
    - threshold: area threshold: to determine which layer to extract from
'''

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import os
from torchvision.utils import save_image


###############################################################################################
#    sub-functions
###############################################################################################

def extract_feature(args, coordinate, feature_outputs, threshold, pool_size):
    x1 = coordinate[0]
    y1 = coordinate[1]
    x2 = coordinate[2]
    y2 = coordinate[3]
    bbox_size = (x2-x1) * (y2-y1)
    if bbox_size < threshold[0]:                # [0]: smaller threshold
        feature_layer = feature_outputs[0]      # [0]: layer 2
    elif bbox_size < threshold[1]:
        feature_layer = feature_outputs[1]
    else:
        feature_layer = feature_outputs[2]

    feature_size = feature_layer.size()
    scaled_coor = scale_bbox(args, coordinate, [feature_size[3],feature_size[2]])
    bbox_feature = feature_layer [:, :, scaled_coor[1]:scaled_coor[3], scaled_coor[0]:scaled_coor[2]]  # torch.Size([1, 1024, 4, 6])
    
    att = F.adaptive_avg_pool2d(bbox_feature,[pool_size[1],pool_size[2]]).permute(0, 2, 3, 1)  # torch.Size([1, 1, 1, 1024])
    att_size = att.size()
    if att_size[3] != pool_size[0]:
        att = F.adaptive_avg_pool2d(att,[pool_size[1],pool_size[0]]).permute(0, 1, 2, 3)  # torch.Size([1, 1, 1, 512])        
    att = torch.flatten(att)  # torch.Size([512])

    ## To check if have 'nan' value in the extracted features due to small bbox area
    feature_sum = np.sum(att.data.cpu().float().numpy())
    array_has_nan = np.isnan(feature_sum )
    if array_has_nan :
        print('!!!!!!!!!!', (x2-x1)*(y2-y1), scaled_coor, (scaled_coor[2]-scaled_coor[0])*(scaled_coor[3]-scaled_coor[1]), bbox_feature.shape)   # bbox_feature = torch.Size([1, 512, 6, 0])
    # elif (x2-x1)*(y2-y1) < 360:
    #     print('#########', (x2-x1)*(y2-y1), scaled_coor, (scaled_coor[2]-scaled_coor[0])*(scaled_coor[3]-scaled_coor[1]), bbox_feature.shape)
    return att


def scale_bbox(args, coordinates, feature_map_size):
    ''' scale the size of the bbox to the size of feature map
    '''
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

    return [int(x1),int(y1),int(x2),int(y2)]

def normalize_gtbbox(args, cls, gt_bbox_list):
    ''' scale the size of GT bbox to the size of image 
    '''
    x1 = int(gt_bbox_list[cls*4])
    y1 = int(gt_bbox_list[(cls*4)+1])
    x2 = int(gt_bbox_list[(cls*4)+2])
    y2 = int(gt_bbox_list[(cls*4)+3])

    x1 = x1 * args.imgw / 1280
    x2 = x2 * args.imgw / 1280
    y1 = y1 * args.imgh / 1024
    y2 = y2 * args.imgh / 1024

    return int(x1),int(y1),int(x2),int(y2)

def compute_iou(boxA, boxB):
    # https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
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
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou


def crop_bbox(coordinate, rgb_img):
    x1,y1,x2,y2 = coordinate[0], coordinate[1], coordinate[2], coordinate[3]
    # print(x,y,w,h)
    crop_region = rgb_img[y1:y2, x1:x2]
    # print(rgb_img.shape)
    # print(crop_image.shape)
    return crop_region
    


def form_bboxes(grey_img, rgb_img, threshold, colour, cls):
        
        # grey_img is between 0 ~ 1
        coordinate_list = []
        ret,thresh = cv2.threshold(grey_img,threshold,255,cv2.THRESH_BINARY)
        thresh = thresh.astype(np.uint8)
        
        # contours retrieve and approximation: https://medium.com/analytics-vidhya/opencv-findcontours-detailed-guide-692ee19eeb18
        # contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
        contours,hierarchy = cv2.findContours(thresh, 1,2)
        
        cv2.imwrite('grey2.png', thresh)
        # if not len(contours) == 0:
        #     print('contours:', len(contours[0]))

        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt) # x, y is the top left corner, and w, h are the width and height respectively            
            bbox_size = w * h
            if bbox_size > 25:
                coordinate_list.append([x,y,x+w,y+h])
                # crop_bbox([x,y,w,h], rgb_img, cls)
                cv2.rectangle(rgb_img, (x,y), (x+w,y+h), colour,2)
        # cv2.imwrite('bbox.png', rgb_img)
        cv2.waitKey()
        return rgb_img, coordinate_list



# BGR
# bipolar_forceps (class 0) => red: (0,0,255)
# prograsp_forceps (class 1) => blue: (255,0,0)
# large_needle_driver (class 2) => green: (0,255,0)
# monopolar_curved_scissors (class 3) => white: (255,255,255)
# ultrasound_probe (class 4) => yellow: (0,255,255)
# suction (class 5) => purple: (255,0,255)
# clip_applier (class 6) => light blue: (255,255,0)
# stapler (class 7) => other green: (116, 139, 69)
# hook (class 8) => brick: (31, 102, 156)
# specimen bag (class 9) => grey: (128, 128, 128)
# tissue (class 10) => orange: (3, 97, 255)

class_colour = [(0,0,255), (255,0,0), (0,255,0), (255,255,255), (0,255,255), (255,0,255), (255,255,0), (116, 139, 69), (31, 102, 156), (147, 20, 255), (3,97,255)]


#############################################################################################
#   main functions
#############################################################################################

def plot_multiplebbox(args, cam, input_tensor, rgb_img, threshold, gt_bbox_list=None):
# def plot_multiplebbox(args, cam, input_tensor, rgb_img, threshold):
    ''' plot multiple bbox for all present classes in the image
    '''
    nor_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    nor_img = 255 * ((nor_img - np.min(nor_img)) / (np.max(nor_img) - np.min(nor_img)))
    nor_img = nor_img.astype(np.uint8)

    save_image_name = 'bbox_' + str(threshold) + '.png'
    coordinate_list_all_cls = []

    for cls in range (args.cls):
        temp_list = [cls]
        grayscale_cam = cam(input_tensor=input_tensor, target_category=cls)  #grayscale_cam: (bs,H,W)
        grayscale_cam = grayscale_cam[args.fidx, :]
        bbox, coordinate_list = form_bboxes(grayscale_cam, nor_img, threshold, class_colour[cls], cls)
        # ## plot for gt bbox
        # if cls <= 8:             
        #     x1,y1,x2,y2 = normalize_gtbbox(args, cls, gt_bbox_list)
        #     cv2.rectangle(nor_img, (x1,y1), (x2,y2), class_colour[cls], 2)
        temp_list.append(coordinate_list)
        coordinate_list_all_cls.append(temp_list)
    cv2.imwrite(save_image_name, bbox)
    cv2.destroyAllWindows()
    return coordinate_list_all_cls



def crop_multibbox(args, rgb_img, coordinate_list_all_cls, frame_path):
    ''' crop and save raw image based on the bbox coordinates 
    '''
    nor_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    nor_img = 255 * ((nor_img - np.min(nor_img)) / (np.max(nor_img) - np.min(nor_img)))
    nor_img = nor_img.astype(np.uint8)

    for cls in range (len(coordinate_list_all_cls)-1):
        coordinate_list = coordinate_list_all_cls[cls][1]
        for i in range (len(coordinate_list)):
            x1 = coordinate_list[i][0]
            y1 = coordinate_list[i][1]
            x2 = coordinate_list[i][2]
            y2 = coordinate_list[i][3]
            crop_region = crop_bbox(coordinate_list[i], nor_img)
            bbox_name = frame_path + '_{}_{}_{},{},{},{}.png'.format(cls, i, x1, y1, x2, y2)
            cv2.imwrite(bbox_name, crop_region)
    
    # for tissue class
    coordinate_list = coordinate_list_all_cls[args.cls-1][1]
    if coordinate_list:       
        x1 = min(x[0] for x in coordinate_list)
        y1 = min(x[1] for x in coordinate_list)
        x2 = max(x[2] for x in coordinate_list)
        y2 = max(x[3] for x in coordinate_list)
        combined_bbox = [x1,y1,x2,y2]
        crop_region = crop_bbox(combined_bbox, nor_img)
        bbox_name = frame_path + '_{}_{}_{},{},{},{}.png'.format(args.cls-1, 0, x1, y1, x2, y2)
        cv2.imwrite(bbox_name, crop_region)
            


def compute_multiiou (args, gt_bbox_list, coordinate_list_all_cls):
    ''' compute the IoU for every class in frame-wise
        4 conditions:   1. both GT_bbox and pred_bbox appear: compute IoU using compute_iou
                        2. pred_bbox appear only: IoU = 0
                        3. GT_bbox appear only: IoU = 0
                        4. both bbox does not appear: IoU = NaN (does not contribute to mIoU)
    '''
    
    iou_list_all = []
    
    # for 9 classes in miccai18 dataset, handle only 8 instrument classes
    for cls in range (8):  
        gt_bbox = normalize_gtbbox(args, cls, gt_bbox_list)
        coordinate_list = coordinate_list_all_cls[cls][1]
        iou_list = []
        
        if coordinate_list:         # case 1 and case 2      
            for i in range (len(coordinate_list)):      # all pred_bbox appear in the same class
                x1,y1,x2,y2 = coordinate_list[i]
                pred_xmin = x1
                pred_ymin = y1
                pred_xmax = x2
                pred_ymax = y2
                pred_bbox = [pred_xmin, pred_ymin, pred_xmax, pred_ymax]
                iou = compute_iou(gt_bbox, pred_bbox)
                iou_list.append(iou)
            iou_list_all.append(np.nanmean(iou_list))
        elif gt_bbox != (0,0,0,0):     # case 3
            print('####', gt_bbox)
            iou_list_all.append(0)
        else:                          # case 4
            iou_list_all.append(np.nan)
        
    
    ## iou computation for tissue, combine all tissue bbox as one big bbox
    gt_bbox = normalize_gtbbox(args, 8, gt_bbox_list)       # gt_bbox_list has only 9 classes
    cls = args.cls - 1    # assume tissue class is at the last
    coordinate_list = coordinate_list_all_cls[cls][1]
    
    
    if coordinate_list:      # case 1 and case 2   
        x1 = min(x[0] for x in coordinate_list)
        y1 = min(x[1] for x in coordinate_list)
        x2 = max(x[2] for x in coordinate_list)
        y2 = max(x[3] for x in coordinate_list)
        pred_bbox = [x1,y1,x2,y2]
    else:       # case 3
        pred_bbox = [0,0,0,0]
        if not gt_bbox:     # case 4
            iou_list_all.append(np.nan)         
    
    iou = compute_iou(gt_bbox, pred_bbox)
    iou_list_all.append(iou) 
    
    print(iou_list_all)
    return(iou_list_all)


def extract_feature_all(args, frame_path_bbox, frame_path_frame, coordinate_list_all_cls, feature_outputs, threshold, pool_size):
    frame_features = []
    for cls in range (len(coordinate_list_all_cls)-1):
        coordinate_list = coordinate_list_all_cls[cls][1]
        for i in range (len(coordinate_list)):
            x1 = coordinate_list[i][0]
            y1 = coordinate_list[i][1]
            x2 = coordinate_list[i][2]
            y2 = coordinate_list[i][3]
            att = extract_feature(args, coordinate_list[i], feature_outputs, threshold, pool_size)
            bbox_name = frame_path_bbox + '_{}_{}_{},{},{},{}'.format(cls, i, x1, y1, x2, y2)
            np.savez_compressed(bbox_name, feat=att.data.cpu().float().numpy())
            frame_features.append(att.data.cpu().float().numpy())
      
    # for tissue class
    coordinate_list = coordinate_list_all_cls[args.cls-1][1]
    if coordinate_list:       
        x1 = min(x[0] for x in coordinate_list)
        y1 = min(x[1] for x in coordinate_list)
        x2 = max(x[2] for x in coordinate_list)
        y2 = max(x[3] for x in coordinate_list)
        combined_bbox = [x1,y1,x2,y2]
        att = extract_feature(args, combined_bbox, feature_outputs, threshold, pool_size)
        bbox_name = frame_path_bbox + '_{}_{}_{},{},{},{}'.format(args.cls-1, 0, x1, y1, x2, y2)
        np.savez_compressed(bbox_name, feat=att.data.cpu().float().numpy())
        frame_features.append(att.data.cpu().float().numpy())
    
    frame_features = np.array(frame_features)  
    # print(frame_path, frame_features.shape)       # (num_bbox, feature_size)
    np.save(frame_path_frame, frame_features)
