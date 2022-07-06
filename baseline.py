
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.init as init
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
from torch.utils.data import Sampler
from PIL import Image, ImageOps
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import pickle
import numpy as np
import argparse
import copy
import random
import numbers
import os


from train_val import train, val


## fix seed to get result reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser(description="GradCAM model training")
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

parser.add_argument("--epoch", default=200, type=int, help="epochs to train and val")
parser.add_argument("--bs", default=170, type=int, help="batch size")
parser.add_argument(
    "--lr",
    default=0.00001,
    type=float,
    help="learning rate for optimizer, default 1e-3",
)
parser.add_argument(
    "--dropout",
    default=False,
    type=bool,
    help="apply dropout to the classification model",
)
parser.add_argument(
    "--work", default=4, type=int, help="num of workers to use, default 4"
)

parser.add_argument("--save", default=True, type=bool, help="save checkpoint")
parser.add_argument(
    "--load", default=False, type=bool, help="load checkpoint to resume training"
)

parser.add_argument(
    "--finetune", default=False, type=bool, help="fine tune from a pre-trained model"
)
parser.add_argument(
    "--pretrain_model",
    default="ResNet50_256,320_170",
    type=str,
    help="pre-trained model name",
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
print("num of epochs   : {:6d}".format(args.epoch))
print("batch size      : {:6d}".format(args.bs))
print("learning rate   : {:.4f}".format(args.lr))
print("num of workers  : {:6d}".format(args.work))
print("dropout           : ", args.dropout)
print("save checkpoint   : ", args.save)
print("load checkpoint   : ", args.load)
print("fine-tune         : ", args.finetune)
print("pre-trained model : ", args.pretrain_model)
print("==============================")


def pil_loader(path):
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


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
    def __init__(self, degrees):
        self.degrees = degrees
        self.count = 0

    def __call__(self, img):
        seed = self.count // sequence_length
        random.seed(seed)
        self.count += 1
        angle = random.randint(-self.degrees, self.degrees)
        return TF.rotate(img, angle)


class ColorJitter(object):
    def __init__(self, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1):
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
        hue_factor = random.uniform(-self.hue, self.hue)

        img_ = TF.adjust_brightness(img, brightness_factor)
        img_ = TF.adjust_contrast(img_, contrast_factor)
        img_ = TF.adjust_saturation(img_, saturation_factor)
        img_ = TF.adjust_hue(img_, hue_factor)

        return img_


class CholecDataset(Dataset):
    """Dataset class for Grad-CAM model
    input: images directory, intruments annotations, transform configurations, image loader
    output: image and label

    note: modified from https://github.com/YuemingJin/TMRNet/blob/main/code/Training%20memory%20bank%20model/train_singlenet_phase_1fc.py
    """

    def __init__(self, file_paths, file_labels, transform=None, loader=pil_loader):
        self.file_paths = file_paths
        self.file_labels_tool = file_labels[:, 0]
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_names = self.file_paths[index]
        labels_tool = self.file_labels_tool[index]
        imgs = self.loader(img_names)
        if self.transform is not None:
            imgs = self.transform(imgs)

        return imgs, labels_tool

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
        x = x.view(
            -1, 3, args.imgh, args.imgw
        )  # [batch_size, seq_len, image_size_H, image_size_W]
        x = self.share.forward(x)  # [batch_size*seq_len, 2048, 1, 1]
        x = x.view(-1, sequence_length, 2048)  # [batch_size, seq_len, 2048]
        self.lstm.flatten_parameters()
        y, _ = self.lstm(x)  # [batch_size, seq_len, 512]
        y = y.contiguous().view(-1, 512)  # [batch_size*seq_len, 512]
        y = self.dropout(y)
        y = self.fc(y)  # [batch_size*seq_len, num_class]
        return y


def get_useful_start_idx(sequence_length, list_each_length):
    """get the start index of every set of the image sequence
    example:
    index = get_useful_start_idx(sequence_length = 3, list_each_length = [4,5])
    idx of all image sequence: [[0,1,2],[1,2,3],[4,5,6],[5,6,7],[6,7,8]]
    index = [0, 1, 4, 5, 6]

    Input: sequence_length (int), number of frames in each sequence
    Output: index of the first frame in each set image sequence
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

    train_transforms = None
    test_transforms = None

    train_transforms = transforms.Compose(
        [
            transforms.Resize((args.imgh, args.imgw)),
            ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            RandomHorizontalFlip(),
            RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.46641618, 0.34214595, 0.36506417],
                [0.20304796, 0.18248262, 0.19647568],
            ),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.Resize((args.imgh, args.imgw)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.46641618, 0.34214595, 0.36506417],
                [0.20304796, 0.18248262, 0.19647568],
            ),
        ]
    )

    train_dataset_40 = CholecDataset(train_paths_40, train_labels_40, train_transforms)
    val_dataset_40 = CholecDataset(val_paths_40, val_labels_40, test_transforms)

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


def train_model(train_dataset, train_num_each, val_dataset, val_num_each):

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

    val_loader = DataLoader(
        val_dataset_40,
        batch_size=args.bs,
        sampler=SeqSampler(val_dataset_40, val_idx),
        num_workers=args.work,
        pin_memory=False,
    )

    """-------------------------------------------------------------------------------------------
                                    Model Selection and Configurations
    ----------------------------------------------------------------------------------------------
    """
    if args.model == "ResNet50":
        model = models.resnet50(pretrained=True)

    elif args.model == "ResNet101":
        model = models.resnet101(pretrained=True)

    elif args.model == "ResNet+LSTM":
        model = resnet_lstm()

    if args.dropout:
        model.fc = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(model.fc.in_features, args.cls)
        )
    else:
        model.fc = nn.Linear(model.fc.in_features, args.cls)

    if args.finetune:
        # To load cholec80 pretrained model
        model.fc = nn.Linear(2048, 11)
        model = DataParallel(model)
        pretrained_model_path = (
            "./best_model_checkpoints/"
            + args.pretrain_model
            + "/"
            + args.pretrain_model
            + "_best_checkpoint.pth.tar"
        )
        model.load_state_dict(torch.load(pretrained_model_path))
        model = model.module
        model.fc = nn.Linear(model.fc.in_features, args.cls)

    if args.multiple_gpu:
        model = DataParallel(model)
        model.to(device)
    else:
        model.to(device)

    # 1. BCELoss plus a Sigmoid function operation will get BCEWithLogitsLoss.
    # 2. MultiLabelSoftMarginLoss and BCEWithLogitsLoss are the same from the formula.
    # https://www.programmersought.com/article/33036452919/#class-torchnnmultilabelsoftmarginlossweightnone-size_averagetruesource
    criterion_tool = nn.MultiLabelSoftMarginLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        nesterov=False,
        weight_decay=0.0001,
    )
    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.95, patience=3, mode="min"
    )

    """--------------------------------------------------------------------------------------------------------
                                    Saving and loading checkpoints to resume training
    -----------------------------------------------------------------------------------------------------------
    """
    result_filename = (
        "miccai2018_11class_cholec_"
        + args.model
        + "_"
        + str(args.imgh)
        + ","
        + str(args.imgw)
        + "_"
        + str(args.bs)
        + "_lr_"
        + str(args.lr)
    )
    save_path = "./best_model_checkpoints/" + result_filename + "/"

    if not os.path.exists(save_path):
        print("The new directory is created:", save_path)
        os.mkdir(save_path)

    print("Save path:", save_path)
    checkpoint_path = save_path + result_filename + "_checkpoint.pt"
    best_checkpoint_path = save_path + result_filename + "_best_checkpoint.pth.tar"

    if args.load == True:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        best_epoch = checkpoint["best_epoch"]
        best_mAP = checkpoint["best_mAP"]
        InfoList = checkpoint["info_list"]

        print(
            "Last training epoch:"
            + str(epoch)
            + "  Last testing loss:"
            + str(InfoList[epoch + 1][6])
        )
        print("Last best epoch:" + str(best_epoch) + "  Last best mAP:" + str(best_mAP))
        epoch += 1
    else:
        best_epoch = 0
        best_mAP = 0.0
        epoch = 0
        InfoList = [
            [
                "epoch",
                "lr",
                "train_mean_loss",
                "train_acc",
                "train_mAP",
                "train_elapsed_time",
                "val_mean_loss",
                "val_acc",
                "val_mAP",
                "val_elapsed_time",
            ]
        ]

    """--------------------------------------------------------------------------------------------------
                                        Training and Validation
    -----------------------------------------------------------------------------------------------------
    """
    while epoch in range(args.epoch):
        torch.cuda.empty_cache()
        np.random.shuffle(train_we_use_start_idx_40)

        train_idx_40 = []
        for i in range(num_train_we_use_40):
            for j in range(sequence_length):
                train_idx_40.append(train_we_use_start_idx_40[i] + j)

        train_loader_40 = DataLoader(
            train_dataset_40,
            batch_size=args.bs,
            sampler=SeqSampler(train_dataset_40, train_idx_40),
            num_workers=args.work,
            pin_memory=False,
        )

        lr = optimizer.param_groups[0]["lr"]
        tempInfo = [epoch, lr]
        trainInfo = train(
            args,
            epoch,
            num_train_all,
            model,
            train_loader_40,
            optimizer,
            criterion_tool,
        )
        valInfo = val(args, epoch, num_val_all, model, val_loader, criterion_tool)
        tempInfo.extend(trainInfo)
        tempInfo.extend(valInfo)
        InfoList.append(tempInfo)

        val_mAP = valInfo[2]
        if val_mAP > best_mAP:
            best_mAP = val_mAP
            best_epoch = epoch
            if args.save == True:
                best_model = copy.deepcopy(model)
                torch.save(best_model.state_dict(), best_checkpoint_path)

        print(
            "epoch: {}  Acc: {:.4f}  mAP: {:.4f}  best epoch: {}  best mAP: {:.4f} val loss: {:.6f} lr: {:.6f}".format(
                epoch,
                valInfo[1],
                valInfo[2],
                best_epoch,
                best_mAP,
                valInfo[0],
                optimizer.param_groups[0]["lr"],
            )
        )

        # save every epoch
        if args.save == True:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_epoch": best_epoch,
                    "best_mAP": best_mAP,
                    "info_list": InfoList,
                },
                checkpoint_path,
            )
        np.savetxt(
            save_path + result_filename + "_info_list.csv",
            InfoList,
            delimiter=", ",
            fmt="% s",
        )

        val_mean_loss = valInfo[0]
        exp_lr_scheduler.step(val_mean_loss)
        epoch += 1


def main():
    seed_everything()
    train_dataset_40, train_num_each_40, val_dataset_40, val_num_each_40 = get_data(
        "./miccai2018_train_val_paths_labels_adjusted.pkl"
    )
    train_model(
        (train_dataset_40), (train_num_each_40), (val_dataset_40), (val_num_each_40)
    )


if __name__ == "__main__":
    main()

print("Done")
print()
