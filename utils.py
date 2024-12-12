import torch
from datetime import datetime
from pathlib import Path
from tqdm.auto import tqdm
import json
from cjm_pytorch_utils.core import pil_to_tensor, tensor_to_pil, get_torch_device, set_seed, denorm_img_tensor, \
    move_data_to_device
from torch.amp import autocast
import math
from functools import partial
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
# Import Mask R-CNN
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, maskrcnn_resnet50_fpn, MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from cjm_pil_utils.core import resize_img, get_img_files, stack_imgs
from torchvision.io import write_png

# 项目名称，可以自行修改，里面会存放一些存放了训练的一些模型信息，如果没有需要自行创建
project_name = f"pytorch-mask-r-cnn-instance-segmentation"
project_dir = Path(f"./{project_name}/")
draw_bboxes = partial(draw_bounding_boxes, fill=False, width=2)


def generate_colors(num_colors):
    """生成足够多的对比色"""
    colormap = cm.get_cmap('plasma', num_colors)  # 使用 'tab20' 颜色映射
    colors = [
        mcolors.to_hex(colormap(i))  # 转换为十六进制颜色字符串
        for i in range(num_colors)
    ]
    return colors


def find_single_json_file(directory: str):
    """
    自动解析目录下唯一的 JSON 文件路径。

    :param directory: 要搜索的目录路径
    :return: 如果存在唯一的 JSON 文件，返回其路径；否则返回 None
    """
    dir_path = Path(directory)
    json_files = list(dir_path.rglob("*.json"))  # 搜索所有 .json 文件

    if len(json_files) == 1:
        return str(json_files[0])  # 返回唯一 JSON 文件路径
    elif len(json_files) > 1:
        raise ValueError("There are multiple JSON files in the directory.")
    else:
        raise FileNotFoundError("No JSON file found in the directory.")


def create_mask_model(class_names: dict) -> MaskRCNN:
    model = maskrcnn_resnet50_fpn_v2(weights='DEFAULT')

    # 得到模型的输入channel
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

    # 得到mask的输出channel
    dim_reduced = model.roi_heads.mask_predictor.conv5_mask.out_channels

    # 替换输入输出的channel
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_features_box, num_classes=len(class_names))
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels=in_features_mask, dim_reduced=dim_reduced,
                                                       num_classes=len(class_names))
    return model


def run_epoch(model, dataloader, optimizer, lr_scheduler, device, scaler, epoch_id, is_training):
    """
    Function to run a single training or evaluation epoch.

    Args:
        model: A PyTorch model to train or evaluate.
        dataloader: A PyTorch DataLoader providing the data.
        optimizer: The optimizer to use for training the model.
        loss_func: The loss function used for training.
        device: The device (CPU or GPU) to run the model on.
        scaler: Gradient scaler for mixed-precision training.
        is_training: Boolean flag indicating whether the model is in training or evaluation mode.

    Returns:
        The average loss for the epoch.
    """
    # Set the model to training mode
    model.train()

    epoch_loss = 0  # Initialize the total loss for this epoch
    progress_bar = tqdm(total=len(dataloader), desc="Train" if is_training else "Eval")  # Initialize a progress bar

    # Loop over the data
    for batch_id, (inputs, targets) in enumerate(dataloader):
        # Move inputs and targets to the specified device
        inputs = torch.stack(inputs).to(device)

        # Forward pass with Automatic Mixed Precision (AMP) context manager
        with autocast(torch.device(device).type):
            if is_training:
                losses = model(inputs.to(device), move_data_to_device(targets, device))
            else:
                with torch.no_grad():
                    losses = model(inputs.to(device), move_data_to_device(targets, device))

            # Compute the loss
            loss = sum([loss for loss in losses.values()])  # Sum up the losses

        # If in training mode, backpropagate the error and update the weights
        if is_training:
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                old_scaler = scaler.get_scale()
                scaler.update()
                new_scaler = scaler.get_scale()
                if new_scaler >= old_scaler:
                    lr_scheduler.step()
            else:
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

            optimizer.zero_grad()

        # Update the total loss
        loss_item = loss.item()
        epoch_loss += loss_item

        # Update the progress bar
        progress_bar_dict = dict(loss=loss_item, avg_loss=epoch_loss / (batch_id + 1))
        if is_training:
            progress_bar_dict.update(lr=lr_scheduler.get_last_lr()[0])
        progress_bar.set_postfix(progress_bar_dict)
        progress_bar.update()

        # If loss is NaN or infinity, stop training
        if is_training:
            stop_training_message = f"Loss is NaN or infinite at epoch {epoch_id}, batch {batch_id}. Stopping training."
            assert not math.isnan(loss_item) and math.isfinite(loss_item), stop_training_message

    # Cleanup and close the progress bar
    progress_bar.close()

    # Return the average loss for this epoch
    return epoch_loss / (batch_id + 1)


def train_loop(model,
               train_dataloader,
               valid_dataloader,
               optimizer,
               lr_scheduler,
               device,
               epochs,
               checkpoint_path,
               use_scaler=False):
    """
    Main training loop.

    Args:
        model: A PyTorch model to train.
        train_dataloader: A PyTorch DataLoader providing the training data.
        valid_dataloader: A PyTorch DataLoader providing the validation data.
        optimizer: The optimizer to use for training the model.
        lr_scheduler: The learning rate scheduler.
        device: The device (CPU or GPU) to run the model on.
        epochs: The number of epochs to train for.
        checkpoint_path: The path where to save the best model checkpoint.
        use_scaler: Whether to scale graidents when using a CUDA device

    Returns:
        None
    """
    # Initialize a gradient scaler for mixed-precision training if the device is a CUDA GPU
    scaler = torch.amp.GradScaler(device='cuda') if device.type == 'cuda' and use_scaler else None
    best_loss = float('inf')  # Initialize the best validation loss

    # Loop over the epochs
    for epoch in tqdm(range(epochs), desc="Epochs"):
        # Run a training epoch and get the training loss
        train_loss = run_epoch(model, train_dataloader, optimizer, lr_scheduler, device, scaler, epoch,
                               is_training=True)
        # Run an evaluation epoch and get the validation loss
        with torch.no_grad():
            valid_loss = run_epoch(model, valid_dataloader, None, None, device, scaler, epoch, is_training=False)

        # If the validation loss is lower than the best validation loss seen so far, save the model checkpoint
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)

        # Save metadata about the training process
        training_metadata = {
            'epoch': epoch,
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'learning_rate': lr_scheduler.get_last_lr()[0],
            'model_architecture': model.name
        }
        with open(Path(checkpoint_path.parent / 'training_metadata.json'), 'a') as f:
            json.dump(training_metadata, f)
            f.write('\n')
    # If the device is a GPU, empty the cache
    if device.type != 'cpu':
        getattr(torch, device.type).empty_cache()


def parse_timestamp(file_name):
    """
    从文件名解析时间戳
    :param file_name: 图片文件名，如 P22033011333910.jpg
    :return: 解析后的时间字符串，如 '2022-03-30 11:33'
    """
    # 检查文件名格式是否正确
    if len(file_name) >= 15 and file_name.startswith("P"):
        year = "20" + file_name[1:3]  # 解析年份
        month = file_name[3:5]  # 解析月份
        day = file_name[5:7]  # 解析日期
        hour = file_name[7:9]  # 解析小时
        minute = file_name[9:11]  # 解析分钟
        return f"{year}-{month}-{day} {hour}:{minute}"
    else:
        raise ValueError(f"Invalid file name format: {file_name}")


def update_tracking_data(tracking_data, track_results, timestamp):
    # 为每个时间戳创建一个新的字典，存储实例的宽度和高度
    if timestamp not in tracking_data:
        tracking_data[timestamp] = {}

    for track in track_results:
        x1, y1, x2, y2, track_id = track
        width = x2 - x1
        height = y2 - y1
        # 将实例的宽度和高度存储到时间戳下
        tracking_data[timestamp][f'track_id_{track_id}'] = {
            'width': width,
            'height': height
        }

    return tracking_data


def parse_timestamp_to_date(timestamp):
    # 假设时间戳格式为 "22022-3.30-11:33" 或类似格式
    date_str = timestamp[:10]  # 提取年份和日期部分
    time_str = timestamp[11:]  # 提取时间部分
    # 转换为标准时间格式
    date_obj = datetime.strptime(f"{date_str}", "%Y-%m-%d")
    time_obj = datetime.strptime(time_str, "%H:%M")
    full_datetime = datetime.combine(date_obj, time_obj.time())
    return full_datetime
