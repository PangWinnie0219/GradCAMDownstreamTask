import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from torchvision.utils import save_image
import time


def train(args, epoch, num_train_all, model, train_loader, optimizer, criterion):
    """input: args, epoch, total number of training images, model, dataloader, optimizer, loss function (criterion)
    output: training information (list): [train_mean_loss, train_acc, train_mAP, train_elapsed_time]
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Sets the module in training mode.
    model.train()
    batch_progress = 0.0
    train_start_time = time.time()

    train_outputs_tool_list = []
    train_labels_tool_list = []
    train_scores_tool_list = []
    train_loss_tool_list = []

    for idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        inputs, labels_tool = data[0].to(device), data[1].to(device)

        if args.method == "temporal":
            ### take every 3th value from the list
            # Before: [batch_size, num_class] =>  torch.Size([300, 7])
            # After: [batch_size/sequence_len, num_class] =>  torch.Size([100, 7])
            labels_tool = labels_tool[(args.seq - 1) :: args.seq]

            # Before: [batch_size, 3, image_size_H, image_size_W] => torch.Size([300, 3, 240, 420])
            # After: [batch_size/sequence_len, sequence_len, 3, image_size_H, image_size_W] => torch.Size([100, 3, 3, 240, 420])
            inputs = inputs.view(-1, args.seq, 3, args.imgh, args.imgw)
            outputs_tool = model.forward(inputs)
            outputs_tool = outputs_tool[args.seq - 1 :: args.seq]

        elif args.method == "non-temporal":
            outputs_tool = model.forward(inputs)

        loss_tool = criterion(outputs_tool, labels_tool.float())
        train_loss_tool_list.append(loss_tool.item())
        loss = loss_tool
        loss.backward()
        optimizer.step()

        # A simple code to show the mAP and accuracy (sample-wise accu, class-wise accu) in Multi-Label Classification case.
        # https://colab.research.google.com/drive/1wrku2Im30VRhTChLJgGbOR4clkBGBscM
        outputs_sigmoid_tool = torch.sigmoid(outputs_tool)
        train_outputs_tool_list.extend(outputs_sigmoid_tool.detach().cpu().numpy())
        train_labels_tool_list.extend(labels_tool.detach().cpu().numpy())

        scores_tool = torch.round(outputs_sigmoid_tool.data)
        train_scores_tool_list.extend(scores_tool.detach().cpu().numpy())

        batch_progress += 1
        if batch_progress * args.bs >= num_train_all:
            percent = 100.0
            print(
                "Batch progress: %s [%d/%d]"
                % (str(percent) + "%", num_train_all, num_train_all),
                end="\n",
            )
        else:
            percent = round(batch_progress * args.bs / num_train_all * 100, 2)
            print(
                "Batch progress: %s [%d/%d]"
                % (str(percent) + "%", batch_progress * args.bs, num_train_all),
                end="\r",
            )

    train_elapsed_time = time.time() - train_start_time

    train_pred = average_precision_score(
        np.array(train_labels_tool_list),
        np.array(train_outputs_tool_list),
        average=None,
    )
    train_mAP = np.nanmean(train_pred)
    train_acc = accuracy_score(
        np.array(train_labels_tool_list), np.array(train_scores_tool_list)
    )
    train_mean_loss = np.mean(np.array(train_loss_tool_list))
    trainInfo = [train_mean_loss, train_acc, train_mAP, train_elapsed_time]

    print(
        "epoch: {:d}"
        "  train in: {:2.0f}m{:2.0f}s"
        "  train loss(tool): {:4.4f}"
        "  train mAP(tool): {:.4f} "
        "  train acc(tool): {:.4f}".format(
            epoch,
            train_elapsed_time // 60,
            train_elapsed_time % 60,
            train_mean_loss,
            train_mAP,
            train_acc,
        )
    )
    return trainInfo


def val(args, epoch, num_val_all, model, val_loader, criterion):
    """input: args, epoch, total number of validation images, model, dataloader, loss function (criterion)
    output: validation information (list): [val_mean_loss, val_acc, val_mAP, val_elapsed_time]
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Sets the module in evaluation mode.
    model.eval()
    val_start_time = time.time()
    val_progress = 0

    val_outputs_tool_list = []
    val_labels_tool_list = []
    val_scores_tool_list = []
    val_loss_tool_list = []

    with torch.no_grad():
        for data in val_loader:
            val_start_time = time.time()
            inputs, labels_tool = data[0].to(device), data[1].to(device)

            if args.method == "temporal":
                labels_tool = labels_tool[(args.seq - 1) :: args.seq]

                inputs = inputs.view(-1, args.seq, 3, args.imgh, args.imgw)
                # save_image(inputs[0], 'img1.png')
                outputs_tool = model.forward(inputs)
                outputs_tool = outputs_tool[args.seq - 1 :: args.seq]
            elif args.method == "non-temporal":
                outputs_tool = model.forward(inputs)

            loss_tool = criterion(outputs_tool, labels_tool.float())
            val_loss_tool_list.append(loss_tool.item())

            outputs_sigmoid_tool = torch.sigmoid(outputs_tool)
            val_outputs_tool_list.extend(outputs_sigmoid_tool.detach().cpu().numpy())
            val_labels_tool_list.extend(labels_tool.detach().cpu().numpy())

            scores_tool = torch.round(outputs_sigmoid_tool.data)
            val_scores_tool_list.extend(scores_tool.detach().cpu().numpy())

            val_progress += 1
            if val_progress * args.bs >= num_val_all:
                percent = 100.0
                print(
                    "Val progress: %s [%d/%d]"
                    % (str(percent) + "%", num_val_all, num_val_all),
                    end="\n",
                )
            else:
                percent = round(val_progress * args.bs / num_val_all * 100, 2)
                print(
                    "Val progress: %s [%d/%d]"
                    % (str(percent) + "%", val_progress * args.bs, num_val_all),
                    end="\r",
                )

    val_elapsed_time = time.time() - val_start_time

    val_pred = average_precision_score(
        np.array(val_labels_tool_list), np.array(val_outputs_tool_list), average=None
    )
    val_mAP = np.nanmean(val_pred)
    val_acc = accuracy_score(
        np.array(val_labels_tool_list), np.array(val_scores_tool_list)
    )
    val_mean_loss = np.mean(np.array(val_loss_tool_list))
    valInfo = [val_mean_loss, val_acc, val_mAP, val_elapsed_time]

    print(
        "epoch: {:d}"
        "  val in: {:2.0f}m{:2.0f}s"
        "  val loss(tool): {:4.4f}"
        "  val mAP(tool): {:.4f} "
        "  val acc(tool): {:.4f}".format(
            epoch,
            val_elapsed_time // 60,
            val_elapsed_time % 60,
            val_mean_loss,
            val_mAP,
            val_acc,
        )
    )
    return valInfo
