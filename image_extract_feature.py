"""
Task            : Extract features based on cropped region images
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"

preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize([0.46641618, 0.34214595, 0.36506417], [0.20304796, 0.18248262, 0.19647568])
    ]
)


class myResnet(nn.Module):
    """Feature extractor class
    input: ResNet model
    output: features from crop images [512]
    """

    def __init__(self, resnet):
        super(myResnet, self).__init__()
        self.resnet = resnet

    def forward(self, img, pool_size=[512, 1, 1]):
        x = img.unsqueeze(0)

        x = self.resnet.conv1(x)  # torch.Size([1, 64, 128, 160])
        x = self.resnet.bn1(x)  # torch.Size([1, 64, 128, 160])
        x = self.resnet.relu(x)  # torch.Size([1, 64, 128, 160])
        x = self.resnet.maxpool(x)  # torch.Size([1, 64, 64, 80])

        x = self.resnet.layer1(x)  # torch.Size([1, 256, 64, 80])
        x = self.resnet.layer2(x)  # torch.Size([1, 512, 32, 40])
        x = self.resnet.layer3(x)  # torch.Size([1, 1024, 16, 20])
        x = self.resnet.layer4(x)  # torch.Size([1, 2048, 8, 10])

        att = F.adaptive_avg_pool2d(x, [pool_size[1], pool_size[2]]).permute(
            0, 2, 3, 1
        )  # torch.Size([1, 1, 1, 1024])

        att_size = att.size()
        if att_size[3] != pool_size[0]:
            att = F.adaptive_avg_pool2d(att, [pool_size[1], pool_size[0]]).permute(
                0, 1, 2, 3
            )  # torch.Size([1, 1, 1, 512])

        att = torch.flatten(att)  # torch.Size([512])

        return att


def seed_everything(seed=123):
    print("=================== set the seed :", seed)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_data_paths_list(data_path):
    """return paths of each cropped images
    input: root directory for cropped images
    """
    paths = []
    for dirpath, dirnames, filenames in os.walk(data_path):
        for file in filenames:
            paths.append(os.path.join(dirpath, file))
        paths.sort()
    return paths


def main(params):
    seed_everything(123)  # full set to make reproducible

    """-------------------------------------------------------------------------------------------
                                    GradCAM model and Configurations
    ----------------------------------------------------------------------------------------------
    """
    ## GC-A
    if args.gc_model == 1:
        model = models.resnet50(pretrained=True)
        model.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.fc.in_features, 9))
        model.to(device)
        result_filename = "miccai2018_9class_ResNet50_256,320_32_lr_0.001_dropout_0.2"

    ## GC-B
    elif args.gc_model == 2:
        model = models.resnet50(pretrained=True)
        model.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.fc.in_features, 9))
        model.to(device)
        result_filename = (
            "miccai2018_9class_cholecResNet50_256,320_32_lr_0.001_dropout_0.2"
        )

    ## GC-C
    elif args.gc_model == 3:
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 11)
        model.to(device)
        result_filename = "miccai2018_11class_cholec_ResNet50_256,320_32_lr_0.001"

    ## GC-D
    elif args.gc_model == 4:
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 11)
        model = nn.DataParallel(model)
        model.to(device)
        result_filename = "combine_miccai18_ResNet50_256,320_170"

    else:
        print("Invalid GradCAM model")

    save_path = "./best_model_checkpoints/" + result_filename + "/"
    best_checkpoint_path = save_path + result_filename + "_best_checkpoint.pth.tar"

    checkpoint = torch.load(best_checkpoint_path)
    model.load_state_dict(checkpoint)

    if isinstance(model, nn.DataParallel):
        model = model.module
    my_resnet = myResnet(model)
    my_resnet.to(device)
    my_resnet.eval()

    image_paths = get_data_paths_list(args.images_root)
    print("Total bbox len:", len(image_paths))

    dir_att = params["output_dir"]
    print(dir_att)
    dir_att_frame = params["output_dir_frame"]
    print(dir_att)

    previous_frame_name = ""

    for i, img_path in enumerate(image_paths):

        ## load the image
        I = Image.open(img_path).convert("RGB")  # image path
        I = preprocess(I)
        I = I.to(device)  # torch.Size([3, 224, 224])

        with torch.no_grad():
            tmp_att = my_resnet(I, params["att_size"])  #  torch.Size([512])

        ## stack all features in the same frame together
        bbox_name = os.path.basename(img_path)
        frame_name = bbox_name[0:8]  # framexxx
        if frame_name != previous_frame_name:
            frame_features = tmp_att.data.cpu().float().numpy()
            frame_features = np.reshape(frame_features, (1, -1))
            print(frame_name)
        else:
            frame_features = np.vstack(
                [frame_features, tmp_att.data.cpu().float().numpy()]
            )

        previous_frame_name = frame_name

        seq_name = os.path.basename(os.path.dirname(img_path))
        feature_path = os.path.join(dir_att, seq_name)
        feature_path_frame = os.path.join(dir_att_frame, seq_name)

        if not os.path.exists(feature_path):
            os.makedirs(feature_path)
        if not os.path.exists(feature_path_frame):
            os.makedirs(feature_path_frame)

        #### write to pkl
        ##  frame_feature
        print(os.path.join(feature_path_frame, frame_name), frame_features.shape)
        np.save(os.path.join(feature_path_frame, frame_name), frame_features)

        ## bbox_feature
        feature_name = str(img_path).split("/")[-1].split(".")[0]
        print(os.path.join(feature_path, feature_name))
        np.savez_compressed(
            os.path.join(feature_path, feature_name),
            feat=tmp_att.data.cpu().float().numpy(),
        )

    print("wrote ", params["output_dir"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # options
    parser.add_argument(
        "--images_root",
        default="/media/mmlab/data_2/winnie/GradCAM_miccai2018/region_image_bbox/model_4_pred0.43_0_0.1_512,1,1/",
        help="root location in which images are stored, to be prepended to file_path in input json",
    )
    parser.add_argument("--att_size", default=[128, 2, 2])
    parser.add_argument(
        "--model", type=str, default="resnet50", help="resnet101, resnet152"
    )
    parser.add_argument(
        "--gc_model",
        type=int,
        default=4,
        help="GradCAM model options: 0(GC-A), 1(GC-B), 2(GC-C), 3(GC-D",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./region_features_bbox_method1/model_4_pred0.43_0_0.1_128,2,2_resize224,224_layer3",
    )
    parser.add_argument(
        "--output_dir_frame",
        type=str,
        default="./region_features_frame_method1/model_4_pred0.43_0_0.1_128,2,2_resize224,224_layer3",
    )

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    main(params)
