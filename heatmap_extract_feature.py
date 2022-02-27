"""
Project         : Model Deployment Using Grad-CAM for Medical Downstream Tasks
Lab             : MMLAB, National University of Singapore
contributors    : Pang Winnie, Sai Mitheran, Xu Mengya, Lalithkumar Seenivasan, Mobarakol Islam, Hongliang Ren

Task            : Extract features based on heatmap 
                1. Feature map layer to be extracted: layer 3 [1024,16,20]
                2. Get the gradcam heatmap based on the predicted class only
                2. Scale the gradcam heatmap output to the feature map size 
                3. feature_map[heatmap>thr]
                4. adaptive_avg_pool2d to [1,512]
                5. flatten the features: [512]
"""

from utils.gradcam_library import GradCAM
from utils.bbox_library import heatmap_extract_feature
import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
from torch.utils.data import Sampler
from torchvision.utils import save_image
from PIL import Image
import pickle
import numpy as np
import argparse
import random
import os
import copy

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

## fix seed to get result reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser(description="GradCAM model")
parser.add_argument(
    "--multiple_gpu", default=True, type=bool, help="use multiple gpu, default True"
)
parser.add_argument(
    "--method",
    default="non-temporal",
    type=str,
    help="data input method, choices: temporal, non-temporal",
)
parser.add_argument(
    "--model",
    default="ResNet50",
    type=str,
    help="multiple choices: ResNet50, DenseNet121, STGCN, LSTM",
)
parser.add_argument(
    "--cls", default=11, type=int, help="number of class in dataset, default 11"
)
parser.add_argument(
    "--seq",
    default=3,
    type=int,
    help="sequence length (applicable for temporal method only), default 3",
)
parser.add_argument(
    "--imgh", default=256, type=int, help="height of image, default 256"
)
parser.add_argument("--imgw", default=320, type=int, help="width of image, default 320")

parser.add_argument("--bs", default=1, type=int, help="batch size")
parser.add_argument(
    "--dropout",
    default=False,
    type=bool,
    help="apply dropout to the classification model",
)
parser.add_argument(
    "--work", default=4, type=int, help="num of workers to use, default 4"
)
parser.add_argument(
    "--gc_model",
    default=3,
    type=int,
    help="GradCAM model options: 0(GC-A), 1(GC-B), 2(GC-C), 3(GC-D",
)

## To get the Grad-CAM for specific frame
parser.add_argument(
    "--bidx", default=0, type=int, help="batch idx of the frame: frame no / bs"
)
parser.add_argument(
    "--tclass", default=8, type=int, help="target class to show the output of Grad-CAM"
)
parser.add_argument(
    "--threshold",
    default=0.3,
    type=float,
    help="threshold to define the boundary of bbox",
)
parser.add_argument(
    "--pred_thr",
    default=0.5,
    type=float,
    help="threshold for convert model prediction to 0 or 1",
)
parser.add_argument(
    "--adp_size",
    default=512,
    type=int,
    help="adaptive average pooling size: 512, 128 or 32",
)
args = parser.parse_args()


device = "cuda" if torch.cuda.is_available() else "cpu"

### to run on multiple gpus
if args.multiple_gpu == True:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    ## check if it is possible to run on multiple gpu
    num_gpu = torch.cuda.device_count()
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    num_gpu = 1


## decide for the sequence length based on method defined
if args.method == "temporal":
    sequence_length = args.seq
elif args.method == "non-temporal":
    sequence_length = 1
else:
    print("Invalid method defined")


print("==============================")
print("model           :", args.model)
print("method          :", args.method)
print("==============================")
print("number of gpu   : {:6d}".format(args.multiple_gpu))
print("number of class : {:6d}".format(args.cls))
print("sequence length : {:6d}".format(sequence_length))
print("image size H    : {:6d}".format(args.imgh))
print("image size W    : {:6d}".format(args.imgw))
print("batch size      : {:6d}".format(args.bs))
print("num of workers  : {:6d}".format(args.work))
print("target class    : {:6d}".format(args.tclass))
print("bbox threshold  : {:4.2f}".format(args.threshold))
print("pred threshold  : {:4.2f}".format(args.pred_thr))
print("dropout         : ", args.dropout)
print("GradCAM model   : ", args.gc_model)
print("==============================")


def pil_loader(path):
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


class miccaiDataset_val(Dataset):
    """Returns image with transform, transformed image without normalization, label
    input: images directory, intruments annotations, transform configurations (with and without normalization), image loader
    output: 2 images (with and without transformation), label
    """

    def __init__(
        self,
        file_paths,
        file_labels,
        transform=None,
        transform_wo_nor=None,
        loader=pil_loader,
    ):
        self.file_paths = file_paths
        self.file_labels_tool = file_labels[:, 0]
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


class resnet_50(nn.Module):
    """
    Model for feature extraction
    input: model
    output: feature map at layer2, 3 and 4

    #### feature map size: (for image size of 256,320)
    # layer 1 [:-5]: torch.Size([1, 256, 64, 80])
    # layer 2 [:-4]: torch.Size([1, 512, 32, 40])
    # layer 3 [:-3]: torch.Size([1, 1024, 16, 20])
    # layer 4 [:-2]: torch.Size([1, 2048, 8, 10])

    #### alternative method:
    # self.features = nn.Sequential(*list(model.children())[:-4])
    # def forward(self, x):
    #     x = self.features(x)
    #     return x
    """

    def __init__(self, model):
        super(resnet_50, self).__init__()
        self.children_list = []
        self.children_list_2 = []
        self.children_list_3 = []
        self.children_list_4 = []
        self.model = model

        for n, c in self.model.named_children():
            self.children_list.append(c)
            if n == "layer2":
                self.children_list_2 = copy.deepcopy(self.children_list)
            elif n == "layer3":
                self.children_list_3 = copy.deepcopy(self.children_list)
            elif n == "layer4":
                self.children_list_4 = copy.deepcopy(self.children_list)
        self.net2 = nn.Sequential(*self.children_list_2)
        self.net3 = nn.Sequential(*self.children_list_3)
        self.net4 = nn.Sequential(*self.children_list_4)

    def forward(self, x):
        x2 = self.net2(x)
        x3 = self.net3(x)
        x4 = self.net4(x)
        return [x2, x3, x4]


def get_useful_start_idx(sequence_length, list_each_length):
    """get the start index of every set of the image sequence
    example:
    index = get_useful_start_idx(sequence_length = 3, list_each_length = [4,5])
    idx of all image sequence: [[0,1,2],[1,2,3],[4,5,6],[5,6,7],[6,7,8]]
    index = [0, 1, 4, 5, 6]

    input: sequence_length (int), number of frames in each sequence
    output: index of the first frame in each set image sequence
    """
    count = 0
    idx = []
    for i in range(len(list_each_length)):
        for j in range(count, count + (list_each_length[i] + 1 - sequence_length)):
            idx.append(j)
        count += list_each_length[i]
    return idx


def get_data(data_path):
    """prepare the data for dataloader
    input: pickle file containing data directory and labels
    output: training dataset, number of train images in each sequence, validation dataset, number of val images in each sequences
    """

    with open(data_path, "rb") as f:
        train_test_paths_labels = pickle.load(f)
    train_paths_40 = train_test_paths_labels[0]
    val_paths_40 = train_test_paths_labels[1]
    train_labels_40 = train_test_paths_labels[2]
    val_labels_40 = train_test_paths_labels[3]
    train_num_each_40 = train_test_paths_labels[4]
    val_num_each_40 = train_test_paths_labels[5]

    train_labels_40 = np.asarray(train_labels_40, dtype=np.int64)
    val_labels_40 = np.asarray(val_labels_40, dtype=np.int64)

    test_transforms = None

    test_transforms = transforms.Compose(
        [
            transforms.Resize((args.imgh, args.imgw)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.4084945, 0.25513682, 0.25353566], [0.22662906, 0.20201652, 0.1962526]
            ),
        ]
    )

    transforms_wo_nor = transforms.Compose(
        [
            transforms.Resize((args.imgh, args.imgw)),
            transforms.ToTensor(),
        ]
    )

    train_dataset_40 = miccaiDataset_val(
        train_paths_40, train_labels_40, test_transforms, transforms_wo_nor
    )
    val_dataset_40 = miccaiDataset_val(
        val_paths_40, val_labels_40, test_transforms, transforms_wo_nor
    )

    return train_dataset_40, train_num_each_40, val_dataset_40, val_num_each_40


class SeqSampler(Sampler):
    """sample the data for dataloader according to the index
    input: data source, index of all frames in every sequence set
    """

    def __init__(self, data_source, idx):
        super().__init__(data_source)
        self.data_source = data_source
        self.idx = idx

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)


def GradCAM_model(train_dataset, train_num_each, val_dataset, val_num_each):
    """Plot, extract and save all the region features based on the GradCAM heatmap
    input: training dataset, number of train images in each sequence, validation dataset, number of val images in each sequences
    """

    (train_dataset_40), (train_num_each_40), (val_dataset_40), (val_num_each_40) = (
        train_dataset,
        train_num_each,
        val_dataset,
        val_num_each,
    )

    """-----------------------------------------------------------------------------------------------
                                    Dataset Preparation
    ---------------------------------------------------------------------------------------------------
    """
    # get the start index of every set of the image sequence
    train_useful_start_idx_40 = get_useful_start_idx(sequence_length, train_num_each_40)
    val_useful_start_idx_40 = get_useful_start_idx(sequence_length, val_num_each_40)

    # number of the image sequence set
    num_train_we_use_40 = len(train_useful_start_idx_40)
    num_val_we_use_40 = len(val_useful_start_idx_40)

    train_we_use_start_idx_40 = train_useful_start_idx_40
    val_we_use_start_idx_40 = val_useful_start_idx_40

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

    print(
        "num train start idx 40: {:6d}".format(len(train_useful_start_idx_40))
    )  # total train frame - [(sequence_len-1)*num_video]
    print(
        "num of all train use: {:6d}".format(num_train_all)
    )  # number of image sequence set * sequence_len
    print("num of all valid use: {:6d}".format(num_val_all))

    train_loader = DataLoader(
        train_dataset_40,
        batch_size=args.bs,
        sampler=SeqSampler(train_dataset_40, train_idx),
        num_workers=args.work,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset_40,
        batch_size=args.bs,
        sampler=SeqSampler(val_dataset_40, val_idx),
        num_workers=args.work,
        pin_memory=False,
    )

    """-------------------------------------------------------------------------------------------
                                    GradCAM model and Configurations
    ----------------------------------------------------------------------------------------------
    """
    ## GC-A
    if args.gc_model == 1:
        model = models.resnet50(pretrained=True)
        model.fc = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(model.fc.in_features, args.cls)
        )
        model.to(device)
        result_filename = "miccai2018_9class_ResNet50_256,320_32_lr_0.001_dropout_0.2"

    ## GC-B
    elif args.gc_model == 2:
        model = models.resnet50(pretrained=True)
        model.fc = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(model.fc.in_features, args.cls)
        )
        model.to(device)
        result_filename = (
            "miccai2018_9class_cholecResNet50_256,320_32_lr_0.001_dropout_0.2"
        )

    ## GC-C
    elif args.gc_model == 3:
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, args.cls)
        model.to(device)
        result_filename = "miccai2018_11class_cholec_ResNet50_256,320_32_lr_0.001"

    ## GC-D
    elif args.gc_model == 4:
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, args.cls)
        model = DataParallel(model)
        model.to(device)
        result_filename = "combine_miccai18_ResNet50_256,320_170"

    else:
        print("Invalid GradCAM model")

    save_path = "./best_model_checkpoints/" + result_filename + "/"
    best_checkpoint_path = save_path + result_filename + "_best_checkpoint.pth.tar"

    checkpoint = torch.load(best_checkpoint_path)
    model.load_state_dict(checkpoint)

    """---------------------------------------------------------------------------------------------
                                            Grad-CAM
    ------------------------------------------------------------------------------------------------                      
    """
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    # feature extraction model
    feature_model = resnet_50(model)

    ## Saved feature path
    ## ./region_features_bbox_method3/   => frame000_cls.npz  (one .npz for one heatmap, multiple .npz file for 1 frame)
    ## ./region_features_frame_mrthod3/  => frame000.npy  (one .npy for one frame, stack all bbox feature in one .npy file)
    ## ./region_features_bbox/modelNo_predThreshold_camThreshold_bboxThreshold_adaptivePoolingSize

    reg_file_root_bbox = (
        "./region_features_bbox_method3/model_{}_pred{}_0_{}_32,4,4_layer4".format(
            args.gc_model, args.pred_thr, args.threshold, args.adp_size
        )
    )  # path to save the the features
    reg_file_root_frame = (
        "./region_features_frame_method3/model_{}_pred{}_0_{}_32,4,4_layer4".format(
            args.gc_model, args.pred_thr, args.threshold, args.adp_size
        )
    )

    if not os.path.exists(reg_file_root_bbox):
        os.mkdir(reg_file_root_bbox)
    if not os.path.exists(reg_file_root_frame):
        os.mkdir(reg_file_root_frame)

    gt_bbox_path = "/media/mmlab/data_2/winnie/GradCAM_miccai2018/gt_bbox/"
    train_video_list = ["2", "3", "4", "6", "7", "9", "10", "11", "12", "14", "15"]
    val_video_list = ["1", "5", "16"]

    model.eval()
    target_layers = [model.layer4[-1]]

    """------------------------------------------------------------------------------------------- 
                                    Training Set 
    -----------------------------------------------------------------------------------------------"""
    vid_no = 0  # counter for the videos
    frame_no = 0  # counter for the frames, increase when reading to new video

    for idx, data in enumerate(train_loader):
        inputs, inputs_wo_nor, labels_tool = (
            data[0].to(device),
            data[1].to(device),
            data[2].to(device),
        )

        # create a new folder for new video, make current_frame_idx = 0
        if idx >= frame_no:
            vid_name = "seq_" + train_video_list[vid_no]
            current_video_path_bbox = os.path.join(reg_file_root_bbox, vid_name)
            current_video_path_frame = os.path.join(reg_file_root_frame, vid_name)
            print("Creating new folder:", current_video_path_bbox)
            os.mkdir(current_video_path_bbox)
            os.mkdir(current_video_path_frame)

            # to get the frame index from gt_bbox lsit
            gt_bbox = np.loadtxt(
                os.path.join(gt_bbox_path, vid_name + ".txt"), dtype=str, delimiter=", "
            )

            current_frame_idx = 0
            frame_no += train_num_each_40[vid_no]
            vid_no += 1

        input_tensor = inputs
        outputs_tool = model.forward(inputs)
        pred = torch.sigmoid(outputs_tool[0]).detach().cpu().numpy()
        score = np.array(pred > args.pred_thr, dtype=float)

        # Construct the CAM object once, and then re-use it on many images:
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

        """------------------ Extracting Region Features ------------------------------------"""
        current_frame_path_bbox = os.path.join(
            current_video_path_bbox, gt_bbox[current_frame_idx][0]
        )
        current_frame_path_frame = os.path.join(
            current_video_path_frame, gt_bbox[current_frame_idx][0]
        )
        print("Saving bbox features to", current_frame_path_frame)

        if args.adp_size == 512:
            pool_size = [512, 1, 1]
        elif args.adp_size == 128:
            pool_size = [128, 2, 2]
        elif args.adp_size == 32:
            pool_size = [32, 4, 4]
        else:
            print("Invalid size for adaptive average pooling")

        feature_outputs = feature_model(inputs)
        heatmap_extract_feature(
            args,
            cam,
            score,
            input_tensor,
            feature_outputs,
            args.threshold,
            current_frame_path_bbox,
            current_frame_path_frame,
            pool_size,
        )

        current_frame_idx += 1

    """------------------------------------------------------------------------------------------- 
                                    Validation Set 
    -----------------------------------------------------------------------------------------------"""
    vid_no = 0  # counter for the videos
    frame_no = 0  # counter for the frames, increase when reading to new video

    for idx, data in enumerate(val_loader):
        inputs, inputs_wo_nor, labels_tool = (
            data[0].to(device),
            data[1].to(device),
            data[2].to(device),
        )

        # create a new folder for new video, make current_frame_idx = 0
        if idx >= frame_no:
            vid_name = "seq_" + val_video_list[vid_no]
            current_video_path_bbox = os.path.join(reg_file_root_bbox, vid_name)
            current_video_path_frame = os.path.join(reg_file_root_frame, vid_name)
            print("Creating new folder:", current_video_path_bbox)
            os.mkdir(current_video_path_bbox)
            os.mkdir(current_video_path_frame)

            # to get the frame index from gt_bbox lsit
            gt_bbox = np.loadtxt(
                os.path.join(gt_bbox_path, vid_name + ".txt"), dtype=str, delimiter=", "
            )

            current_frame_idx = 0
            frame_no += val_num_each_40[vid_no]
            vid_no += 1

        input_tensor = inputs
        outputs_tool = model.forward(inputs)
        pred = torch.sigmoid(outputs_tool[0]).detach().cpu().numpy()
        score = np.array(pred > args.pred_thr, dtype=float)

        # Construct the CAM object once, and then re-use it on many images:
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

        """------------------ Extracting Region Features ------------------------------------"""
        current_frame_path_bbox = os.path.join(
            current_video_path_bbox, gt_bbox[current_frame_idx][0]
        )
        current_frame_path_frame = os.path.join(
            current_video_path_frame, gt_bbox[current_frame_idx][0]
        )
        print("Saving bbox features to", current_frame_path_frame)

        if args.adp_size == 512:
            pool_size = [512, 1, 1]
        elif args.adp_size == 128:
            pool_size = [128, 2, 2]
        elif args.adp_size == 32:
            pool_size = [32, 4, 4]
        else:
            print("Invalid size for adaptive average pooling")

        feature_outputs = feature_model(inputs)
        heatmap_extract_feature(
            args,
            cam,
            score,
            input_tensor,
            feature_outputs,
            args.threshold,
            current_frame_path_bbox,
            current_frame_path_frame,
            pool_size,
        )

        current_frame_idx += 1


def main():
    seed_everything()
    train_dataset_40, train_num_each_40, val_dataset_40, val_num_each_40 = get_data(
        "./miccai2018_train_val_paths_labels_adjusted.pkl"
    )
    GradCAM_model(
        (train_dataset_40), (train_num_each_40), (val_dataset_40), (val_num_each_40)
    )


if __name__ == "__main__":
    main()

print("Done")
print()
