'''
Task: get rectangle coordinates and show boundary box based on gradcam output
BBOX code modified from https://github.com/zalkikar/BBOX_GradCAM/blob/master/BBOXES_from_GRADCAM.py
'''

import cv2
import numpy as np

def form_bboxes(grey_img, rgb_img, threshold, colour, cls):
        ret,thresh = cv2.threshold(grey_img,threshold,255,cv2.THRESH_BINARY)
        thresh = thresh.astype(np.uint8)
        
        # contours retrieve and approximation: https://medium.com/analytics-vidhya/opencv-findcontours-detailed-guide-692ee19eeb18
        # contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
        contours,hierarchy = cv2.findContours(thresh, 1,2)
        
        cv2.imwrite('grey1.png', thresh)
        # if not len(contours) == 0:
        #     print('contours:', len(contours[0]))

        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt) # x, y is the top left corner, and w, h are the width and height respectively
            cv2.rectangle(rgb_img, (x,y), (x+w,y+h), colour,2)
            # crop image
            # print(x,y,w,h)
            # crop_image = rgb_img[y:y+h, x:x+w]
            # print(rgb_img.shape)
            # print(crop_image.shape)
            # cv2.imwrite('crop_image_{}.png'.format(cls), crop_image)
        cv2.imwrite('bbox.png', rgb_img)
        cv2.waitKey()
        return rgb_img



# BGR
# grasper (class 0) => red: (0,0,255)
# bipolar (class 1) => blue: (255,0,0)
# hook (class 2) => green: (0,255,0)
# scissor (class 3) => white: (255,255,255)
# clipper (class 4) => yellow: (0,255,255)
# irrigator (class 5) => purple: (255,0,255)
# specimen bag (class 6) => light blue: (255,255,0)


class_colour = [(0,0,255), (255,0,0), (0,255,0), (255,255,255), (0,255,255), (255,0,255), (255,255,0), (3,97,255)]

def plot_multiplebbox(args, cam, input_tensor, rgb_img, threshold):
    ''' plot multiple bbox for all present classes in the image
    '''
    nor_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    nor_img = 255 * ((nor_img - np.min(nor_img)) / (np.max(nor_img) - np.min(nor_img)))
    nor_img = nor_img.astype(np.uint8)

    save_image_name = 'bbox_' + str(threshold) + '.png'
    for cls in range (args.cls):
        grayscale_cam = cam(input_tensor=input_tensor, target_category=cls)  #grayscale_cam: (bs,H,W)
        grayscale_cam = grayscale_cam[args.fidx, :]
        bbox = form_bboxes(grayscale_cam, nor_img, threshold, class_colour[cls], cls)
    cv2.imwrite(save_image_name, bbox)
    cv2.destroyAllWindows()

            
