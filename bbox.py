'''Code modified from https://github.com/YuemingJin/TMRNet/blob/main/code/Training%20memory%20bank%20model/train_singlenet_phase_1fc.py
Modification: Grad-CAM
'''

from gradcam_library import GradCAM, GradCAMPlusPlus  # to study for the code only 
# from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, XGradCAM
# from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.image import show_cam_on_image
from bbox_library import plot_multiplebbox
import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
from torch.utils.data import Sampler
from torchvision.utils import save_image
from PIL import Image, ImageOps
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import pickle
import cv2
import numpy as np
import argparse
import random
import numbers
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'

## fix seed to get result reproducibility
def seed_everything(seed=42):
  random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description='lstm training')
parser.add_argument('--multiple_gpu',    default=True,       type=bool, help='use multiple gpu, default True')
parser.add_argument('--method', default='non-temporal', type=str,  help='data input method, choices: temporal, non-temporal') 
parser.add_argument('--model',  default='ResNet50', type=str,  help='multiple choices: ResNet50, ResNet+LSTM, STGCN, LSTM') 
parser.add_argument('--cls',    default=8,          type=int)
parser.add_argument('--seq',    default=3,          type=int,  help='sequence length, default 3')
parser.add_argument('--imgh',   default=120,        type=int,  help='height of image, default 120')
parser.add_argument('--imgw',   default=210,        type=int,  help='width of image, default 210')
parser.add_argument('--bs',     default=81,    type=int,   help='batch size')
parser.add_argument('--work',   default=4,      type=int,   help='num of workers to use, default 4')
parser.add_argument('--load',   type=bool,   default=True,   help='load checkpoint')

## To get the Grad-CAM for specific frame
parser.add_argument('--bidx',     default=0,    type=int,   help='batch idx of the frame: frame no / bs')
parser.add_argument('--fidx',     default=0,    type=int,   help='frame idx of the frame: frame no (remainder) bs')
parser.add_argument('--tclass',   default=7,    type=int,   help='target class to show the output of Grad-CAM')
parser.add_argument('--threshold', default=0.7,    type=float,   help='threshold for defining the area of bbox')
args = parser.parse_args()

mul_gpu = args.multiple_gpu
method = args.method 
model_used = args.model
num_class = args.cls
sequence_length = args.seq
image_size_H = args.imgh
image_size_W = args.imgw
batch_size = args.bs
load_checkpoint = args.load
workers = args.work
frame_idx = args.fidx
batch_idx = args.bidx


# use_gpu = (torch.cuda.is_available() and gpu_usg)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

### to run on multiple gpus
if mul_gpu == True:
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'
    ## check if it is possible to run on multiple gpu
    num_gpu = torch.cuda.device_count()
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    num_gpu = 1


## decide for the sequence length based on method defined
if method == 'temporal':
    sequence_length = sequence_length
elif method == 'non-temporal':
    sequence_length = 1
else:
    print('Invalid method defined')

print('==============================')
print('model           :', model_used)
print('method          :', method)
print('==============================')
print('number of gpu   : {:6d}'.format(num_gpu))
print('number of class : {:6d}'.format(num_class))
print('sequence length : {:6d}'.format(sequence_length))
print('image size H    : {:6d}'.format(image_size_H))
print('image size W    : {:6d}'.format(image_size_W))
print('batch index     : {:6d}'.format(batch_idx))
print('frame index     : {:6d}'.format(frame_idx))
print('threshold       : {:.2f}'.format(args.threshold))
print('batch size      : {:6d}'.format(batch_size))
print('num of workers  : {:6d}'.format(workers))
print('load checkpoint : ', load_checkpoint)
print('==============================')


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class RandomCrop(object):

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.count = 0

    def __call__(self, img):

        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)

        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img

        random.seed(self.count // sequence_length)
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        # print(self.count, x1, y1)
        self.count += 1
        return img.crop((x1, y1, x1 + tw, y1 + th))


class RandomHorizontalFlip(object):
    def __init__(self):
        self.count = 0

    def __call__(self, img):
        seed = self.count // sequence_length
        random.seed(seed)
        prob = random.random()
        self.count += 1
        # print(self.count, seed, prob)
        if prob < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

class RandomRotation(object):
    def __init__(self,degrees):
        self.degrees = degrees
        self.count = 0

    def __call__(self, img):
        seed = self.count // sequence_length
        random.seed(seed)
        self.count += 1
        angle = random.randint(-self.degrees,self.degrees)
        return TF.rotate(img, angle)

class ColorJitter(object):
    def __init__(self,brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.count = 0

    def __call__(self, img):
        seed = self.count // sequence_length
        random.seed(seed)
        self.count += 1
        brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
        contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)
        saturation_factor = random.uniform(1 - self.saturation, 1 + self.saturation)
        hue_factor = random.uniform(- self.hue, self.hue)

        img_ = TF.adjust_brightness(img,brightness_factor)
        img_ = TF.adjust_contrast(img_,contrast_factor)
        img_ = TF.adjust_saturation(img_,saturation_factor)
        img_ = TF.adjust_hue(img_,hue_factor)
        
        return img_


class CholecDataset_1(Dataset):
    ''' returns image with transform, iname with transform without normalization, label
    '''
    
    def __init__(self, file_paths, file_labels, transform=None, transform_wo_nor=None,
                 loader=pil_loader):
        self.file_paths = file_paths
        self.file_labels_tool = file_labels[:,0]
        self.transform = transform
        self.transform_wo_nor = transform_wo_nor
        self.loader = loader

    def __getitem__(self, index):
        img_names = self.file_paths[index]
        labels_tool = self.file_labels_tool[index]
        imgs = self.loader(img_names)
        if self.transform is not None:
            imgs_transform = self.transform(imgs)
        
        # transform without normalization, use to combine with heatmap to show grad-cam results
        if self.transform_wo_nor is not None:
            imgs_transform_wo_nor = self.transform_wo_nor(imgs)
        return imgs_transform, imgs_transform_wo_nor, labels_tool

    def __len__(self):
        return len(self.file_paths)

class resnet_lstm(torch.nn.Module):
    def __init__(self):
        super(resnet_lstm, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.share = torch.nn.Sequential()
        self.share.add_module("conv1", resnet.conv1)
        self.share.add_module("bn1", resnet.bn1)
        self.share.add_module("relu", resnet.relu)
        self.share.add_module("maxpool", resnet.maxpool)
        self.share.add_module("layer1", resnet.layer1)
        self.share.add_module("layer2", resnet.layer2)
        self.share.add_module("layer3", resnet.layer3)
        self.share.add_module("layer4", resnet.layer4)
        self.share.add_module("avgpool", resnet.avgpool)
        self.lstm = nn.LSTM(2048, 512, batch_first=True)
        self.fc = nn.Linear(512, args.cls)
        self.dropout = nn.Dropout(p=0.2)

        init.xavier_normal_(self.lstm.all_weights[0][0])
        init.xavier_normal_(self.lstm.all_weights[0][1])
        init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        x = x.view(-1, 3, image_size_H, image_size_W)   # [batch_size, seq_len, image_size_H, image_size_W]
        x = self.share.forward(x)                       # [batch_size*seq_len, 2048, 1, 1] 
        x = x.view(-1, sequence_length, 2048)           # [batch_size, seq_len, 2048]
        self.lstm.flatten_parameters()
        y, _ = self.lstm(x)                             # [batch_size, seq_len, 512]
        y = y.contiguous().view(-1, 512)                # [batch_size*seq_len, 512]
        y = self.dropout(y)
        y = self.fc(y)                                  # [batch_size*seq_len, num_class]
        return y


def get_useful_start_idx(sequence_length, list_each_length):
    ''' get the start index of every set of the image sequence
        example: 
        index = get_useful_start_idx(sequence_length = 3, list_each_length = [4,5])
        idx of all image sequence: [[0,1,2],[1,2,3],[4,5,6],[5,6,7],[6,7,8]]
        index = [0, 1, 4, 5, 6]
    '''   
    count = 0
    idx = []
    for i in range(len(list_each_length)):
        for j in range(count, count + (list_each_length[i] + 1 - sequence_length)):
            idx.append(j)
        count += list_each_length[i]
    return idx


def get_data(data_path):
    with open(data_path, 'rb') as f:
        train_test_paths_labels = pickle.load(f)
    train_paths_40 = train_test_paths_labels[0]
    val_paths_40 = train_test_paths_labels[1]
    train_labels_40 = train_test_paths_labels[2]
    val_labels_40 = train_test_paths_labels[3]
    train_num_each_40 = train_test_paths_labels[4]
    val_num_each_40 = train_test_paths_labels[5]

    # print('train_paths_40  : {:6d}'.format(len(train_paths_40)))
    # print('train_labels_40 : {:6d}'.format(len(train_labels_40)))
    # print('valid_paths_40  : {:6d}'.format(len(val_paths_40)))
    # print('valid_labels_40 : {:6d}'.format(len(val_labels_40)))

    train_labels_40 = np.asarray(train_labels_40, dtype=np.int64)
    val_labels_40 = np.asarray(val_labels_40, dtype=np.int64)

    train_transforms = None
    test_transforms = None

    train_transforms = transforms.Compose([
            transforms.Resize((image_size_H, image_size_W)),
            # RandomCrop(224),
            ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            RandomHorizontalFlip(),
            RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize([0.4084945, 0.25513682, 0.25353566], [0.22662906, 0.20201652, 0.1962526 ])
        ])

    test_transforms = transforms.Compose([
            transforms.Resize((image_size_H, image_size_W)),
            transforms.ToTensor(),
            transforms.Normalize([0.4084945, 0.25513682, 0.25353566], [0.22662906, 0.20201652, 0.1962526 ])
        ])
    
    transforms_wo_nor = transforms.Compose([
            transforms.Resize((image_size_H, image_size_W)),
            transforms.ToTensor(),
        ])

    train_dataset_40 = CholecDataset_1(train_paths_40, train_labels_40, train_transforms, transforms_wo_nor)
    val_dataset_40 = CholecDataset_1(val_paths_40, val_labels_40, test_transforms, transforms_wo_nor)

    return train_dataset_40, train_num_each_40, \
           val_dataset_40, val_num_each_40



# 序列采样sampler
class SeqSampler(Sampler):
    def __init__(self, data_source, idx):
        super().__init__(data_source)
        self.data_source = data_source
        self.idx = idx

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)


def train_model(train_dataset, train_num_each, val_dataset, val_num_each):

    (train_dataset_40),\
    (train_num_each_40),\
    (val_dataset_40),\
    (val_num_each_40) = train_dataset, train_num_each, val_dataset, val_num_each

    # get the start index of every set of the image sequence
    train_useful_start_idx_40 = get_useful_start_idx(sequence_length, train_num_each_40)  
    val_useful_start_idx_40 = get_useful_start_idx(sequence_length, val_num_each_40)

    # number of the image sequence set
    num_train_we_use_40 = len(train_useful_start_idx_40)
    num_val_we_use_40 = len(val_useful_start_idx_40)

    train_we_use_start_idx_40 = train_useful_start_idx_40
    val_we_use_start_idx_40 = val_useful_start_idx_40

    # np.random.seed(0)
    # np.random.shuffle(train_we_use_start_idx)
    
    # get all index of every element in image sequence set
    # example: [0, 1, 2, 1, 2, 3, 4, 5, 6, 5, 6, 7, 6, 7, 8]
    train_idx = []
    for i in range(num_train_we_use_40):
        for j in range(sequence_length):
            train_idx.append(train_we_use_start_idx_40[i] + j)

    val_idx = []
    for i in range(num_val_we_use_40):
        for j in range(sequence_length):
            val_idx.append(val_we_use_start_idx_40[i] + j)

    num_train_all = len(train_idx)
    num_val_all = len(val_idx)

    print('num train start idx 40: {:6d}'.format(len(train_useful_start_idx_40)))  # total train frame - [(sequence_len-1)*num_video]
    print('num of all train use: {:6d}'.format(num_train_all))  # number of image sequence set * sequence_len
    print('num of all valid use: {:6d}'.format(num_val_all))

    val_loader = DataLoader(
        val_dataset_40,
        batch_size=batch_size,
        sampler=SeqSampler(val_dataset_40, val_idx),
        num_workers=workers,
        pin_memory=False
    )

    if model_used == 'ResNet50':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(2048, num_class)
    elif model_used == 'ResNet+LSTM':
        model = resnet_lstm()
       
    if mul_gpu:
        model = DataParallel(model)
        model.to(device)
    else:
        model.to(device)

    # Load the information from last training session
    if model_used == 'ResNet50':
        # result_filename = model_used + '_' + str(image_size_H) + ',' + str(image_size_W) + '_' + '510'
        result_filename = '8class_' + model_used + '_' + str(image_size_H) + ',' + str(image_size_W) + '_' + '510' 
    elif model_used == 'ResNet+LSTM':
        result_filename = 'LSTM' + '_' + str(image_size_H) + ',' + str(image_size_W) + '_' + '510' + '_Full'

    save_path               = './best_model_checkpoints/'+ result_filename + '/'
    checkpoint_path         = save_path + result_filename + '_checkpoint.pt'

    if load_checkpoint == True:
        checkpoint      = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch           = checkpoint['epoch']
        best_epoch      = checkpoint['best_epoch']
        best_mAP        = checkpoint['best_mAP']
        InfoList        = checkpoint['info_list']
                
        print('Last training epoch:' + str(epoch) + '  Last testing loss:' + str(InfoList[epoch+1][6]))
        print('Last best epoch:' + str(best_epoch) + '  Last best mAP:' + str(best_mAP)) 
        epoch += 1
    else:
        best_epoch = 0
        best_mAP    = 0.0
        epoch       = 0
        InfoList = [['epoch', 'lr', 'train_mean_loss', 'train_acc', 'train_mAP', 'train_elapsed_time', 'val_mean_loss', 'val_acc', 'val_mAP', 'val_elapsed_time']]

    ##### Gad-CAM
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    
    # # Print model's state_dict
    # print("Model's state_dict:")
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    model.eval()
    if model_used == 'ResNet50':
        target_layers = [model.layer4[-1]]
        
        for idx, data in enumerate(val_loader):
            inputs, inputs_wo_nor, labels_tool = data[0].to(device), data[1].to(device), data[2].to(device)
            if idx >= args.bidx:
                outputs_tool = model.forward(inputs)
                input_tensor = inputs
                
                rgb_img = inputs_wo_nor[args.fidx]
                save_image(input_tensor[args.fidx], 'img0.png')   # Normalized image
                save_image(rgb_img, 'img1.png')                   # RGB image
                rgb_img = rgb_img.detach().cpu().numpy()
                rgb_img = rgb_img.transpose(1,2,0)

                print('inputs:', inputs.size())
                print('pred:', torch.sigmoid(outputs_tool[args.fidx]).detach().cpu().numpy())
                print('score:', torch.round(torch.sigmoid(outputs_tool[args.fidx])).detach().cpu().numpy())
                print('label:', labels_tool[args.fidx].detach().cpu().numpy())

                # Construct the CAM object once, and then re-use it on many images:
                cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
                # cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=True)

                # BBox plotting
                plot_multiplebbox(args, cam, input_tensor, rgb_img, args.threshold)
                
                # If None, returns the map for the highest scoring category.
                # Otherwise, targets the requested category.
                #  [grasper, bipolar, hook, scissor, clipper, irrigator, specimen bag]
                if args.tclass == args.cls:
                    target_category = None
                elif args.tclass <  args.cls and args.tclass >= 0:
                    target_category = args.tclass
                    print('target_category:', target_category)
                else:
                    print('Invalid target category')

                # grad-CAM plotting
                grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)  #grayscale_cam: (bs,H,W)
                grayscale_cam = grayscale_cam[args.fidx, :]
                visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                im = Image.fromarray(visualization)
                image_name = "img_" + str(args.tclass) + ".jpeg"
                im.save(image_name)
                break
        
    elif model_used == 'ResNet+LSTM':
        target_layers = [model.share.layer4[-1]]
    
        for idx, data in enumerate(val_loader):
                inputs, inputs_wo_nor, labels_tool = data[0].to(device), data[1].to(device), data[2].to(device)

                if idx >= args.bidx:
                    input_tensor = inputs
                    
                    labels_tool = labels_tool[(args.seq - 1)::args.seq]
                    inputs = inputs.view(-1, args.seq, 3, args.imgh, args.imgw)
                    inputs_wo_nor = inputs_wo_nor.view(-1, args.seq, 3, args.imgh, args.imgw)
                    outputs_tool = model.forward(inputs) 
                    outputs_tool = outputs_tool[args.seq - 1::args.seq]
                    torch.backends.cudnn.enabled=False

                    print('inputs:', inputs.size())
                    print('pred:', torch.sigmoid(outputs_tool[args.fidx]))
                    print('score:', torch.round(torch.sigmoid(outputs_tool[args.fidx])))
                    print('label:', labels_tool[args.fidx])
                    
                    # Construct the CAM object once, and then re-use it on many images:
                    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
                    # cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=True)

                    # If None, returns the map for the highest scoring category.
                    # Otherwise, targets the requested category.
                    #  [grasper, bipolar, hook, scissor, clipper, irrigator, specimen bag]
                    if args.tclass == args.cls:
                        target_category = None
                    elif args.tclass <  args.cls and args.tclass >= 0:
                        target_category = args.tclass
                        print('target_category:', target_category)
                    else:
                        print('Invalid target category')
                    
                    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
                    grayscale_cam = grayscale_cam.reshape(-1, args.seq, args.imgh, args.imgw)  # (bs/num_seq, num_seq, H, W)
                    grayscale_cam_set = grayscale_cam[args.fidx]  # (num_seq, H, W)
                    rgb_img_set = inputs_wo_nor[args.fidx]
                    save_image(rgb_img_set, 'img1.png')                   # RGB image
                    rgb_img_set = rgb_img_set.detach().cpu().numpy()
                    
                    ## Overlay the img and mask frame by frame
                    visualization_array = np.zeros((args.imgh, args.imgw * args.seq, 3))   # (H, W*num_sed, 3) => stack images horizontally
                    for i in range (args.seq):
                        grayscale_cam_frame = grayscale_cam_set[i, :]  # (H,W) 
                        rgb_img = rgb_img_set[i]
                        rgb_img = rgb_img.transpose(1,2,0)   # (H, W, 3)
                        visualization = show_cam_on_image(rgb_img, grayscale_cam_frame, use_rgb=True)  # (H, W, 3)
                        visualization_array[:,args.imgw*i:args.imgw*(i+1),:] = visualization
                    im = Image.fromarray(np.uint8(visualization_array))
                    image_name = "img_" + str(args.tclass) + ".jpeg"
                    im.save(image_name)
                    break

  
def main():
    seed_everything()
    train_dataset_40, train_num_each_40, \
    val_dataset_40, val_num_each_40 = get_data('./train_val_paths_labels_1.pkl')  
    train_model((train_dataset_40),
                (train_num_each_40),
                (val_dataset_40),
                (val_num_each_40))


if __name__ == "__main__":
    main()

print('Done')
print()