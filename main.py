import argparse
import os
import numpy as np
from bambooData import *
from utils import *
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sort import Sort
import json

torchvision.disable_beta_transforms_warning()

tracker = Sort()
device = get_torch_device()
dtype = torch.float32
class_names = ['background', 'bamboo']
FONT_PATH = r"C:\Windows\Fonts\simhei.ttf"  # 替换为你的字体文件路径


def load_model(args):
    """加载模型"""
    model = create_mask_model(class_names)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.eval()
    model.to(device)
    return model


def preprocess_image(img_path, transform):
    """加载并预处理图像"""
    test_img = Image.open(img_path).convert('RGB')
    input_tensor = transform(test_img)[None].to(device)
    return test_img, input_tensor


def get_detections(model, input_tensor, threshold):
    """从模型输出中获取检测结果"""
    with torch.no_grad():
        model_output = model(input_tensor)
    model_output = move_data_to_device(model_output, 'cpu')

    height, width = input_tensor.shape[2], input_tensor.shape[3]
    # 根据分数阈值筛选目标
    scores_mask = model_output[0]['scores'] > threshold
    pred_bboxes = model_output[0]['boxes'][scores_mask]  # 边界框
    pred_labels = [class_names[int(label)] for label in model_output[0]['labels'][scores_mask]]
    pred_scores = model_output[0]['scores'][scores_mask]  # 分数
    pred_masks = torch.concat(
        [torch.where(mask >= threshold, 1, 0) for mask in
         F.interpolate(model_output[0]['masks'][scores_mask], size=(height, width))]).bool()
    return pred_bboxes, pred_labels, pred_scores, pred_masks


def update_tracker(pred_bboxes, pred_scores):
    """更新 SORT 追踪器并返回追踪结果"""
    # 转换为 SORT 需要的格式：[x1, y1, x2, y2, score]
    sort_detections = np.array([
        [box[0].item(), box[1].item(), box[2].item(), box[3].item(), score.item()]
        for box, score in zip(pred_bboxes, pred_scores)
    ])
    return tracker.update(sort_detections)


def annotate_image(test_img, track_results, pred_bboxes, pred_masks, font_path):
    """标注图像"""
    img_tensor = transforms.PILToTensor()(test_img)

    # 获取追踪实例的颜色列表
    colors = generate_colors(len(track_results))

    # 遍历每个追踪目标，绘制分割掩码和边界框
    for i, track in enumerate(track_results):
        x1, y1, x2, y2, track_id = track

        # 找到当前跟踪框对应的掩码
        mask_idx = np.argmin(np.linalg.norm(pred_bboxes.numpy()[:, :4] - np.array([x1, y1, x2, y2]), axis=1))
        current_mask = pred_masks[mask_idx]

        # 在图像上绘制分割掩码
        img_tensor = draw_segmentation_masks(
            image=img_tensor,
            masks=current_mask[None],  # 当前掩码
            alpha=0.5,
            colors=[colors[i % len(colors)]]
        )

        # 在图像上绘制边界框和 ID
        img_tensor = draw_bounding_boxes(
            image=img_tensor,
            boxes=torch.tensor([[x1, y1, x2, y2]]),
            labels=[f"  {int(track_id)}"],
            colors=[colors[i % len(colors)]],
            width=2,
            font=font_path,
            font_size=24
        )
    return img_tensor


def save_image(img_tensor, output_path, file_name):
    """保存标注后的图像"""
    output_file_path = os.path.join(output_path, f"tracked_{file_name.split('.')[0]}.png")
    write_png(img_tensor, output_file_path)
    print(f"Saved tracked prediction for {file_name} to {output_file_path}")


def plot_tracking_data(json_dir):
    json_path = find_single_json_file(json_dir)
    with open(json_path, 'r') as f:
        tracking_data = json.load(f)
    rows = []
    for timestamp, timestamp_data in tracking_data.items():
        for track_id, dimensions in timestamp_data.items():
            row = {'timestamp': timestamp, 'track_id': track_id,
                   'width': dimensions['width'], 'height': dimensions['height']}
            rows.append(row)
    df = pd.DataFrame(rows)
    df['timestamp'] = pd.to_datetime(df['timestamp'].apply(parse_timestamp_to_date))

    # 异常数据处理
    track_counts = df.groupby('track_id')['timestamp'].nunique()
    # 如果某个竹笋在2/3的图片都没检测到，那就别追踪了
    valid_ids = track_counts[track_counts >= len(tracking_data) / 3].index
    df = df[df['track_id'].isin(valid_ids)]

    # 绘制图片
    plt.figure(figsize=(12, 6))
    for track_id in df['track_id'].unique():
        track_data = df[df['track_id'] == track_id]
        plt.plot(track_data['timestamp'], track_data['width'], label=f'Track ID {track_id}')
    plt.xlabel('Time')
    plt.ylabel('Width (px)')
    plt.title('Width Changes Over Time for All Track IDs')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(json_dir, 'width_changes.png'))  # 保存宽度变化图
    plt.close()

    plt.figure(figsize=(12, 6))
    for track_id in df['track_id'].unique():
        track_data = df[df['track_id'] == track_id]
        plt.plot(track_data['timestamp'], track_data['height'], label=f'Track ID {track_id}')
    plt.xlabel('Time')
    plt.ylabel('Height (px)')
    plt.title('Height Changes Over Time for All Track IDs')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(json_dir, 'height_changes.png'))  # 保存高度变化图
    plt.close()


def train(args: argparse):
    dataset_path = args.dataset_path
    annotation_file_paths = find_single_json_file(dataset_path)
    img_dict, annotation_df = get_Img_Ann(dataset_path, annotation_file_paths)
    train_dataset, valid_dataset = getDatasets(img_dict, annotation_df)

    # 构建训练的数据集
    data_loader_params = {
        'batch_size': 4,  # Batch size for data loading
        'num_workers': 8,  # Number of subprocesses to use for data loading
        'persistent_workers': True,
        # If True, the data loader will not shutdown the worker processes after a dataset has been consumed once. This allows to maintain the worker dataset instances alive.
        'pin_memory': 'cuda' in device,
        # If True, the data loader will copy Tensors into CUDA pinned memory before returning them. Useful when using GPU.
        'pin_memory_device': device if 'cuda' in device else '',
        # Specifies the device where the data should be loaded. Commonly set to use the GPU.
        'collate_fn': lambda batch: tuple(zip(*batch)),
    }
    train_dataloader = DataLoader(train_dataset, **data_loader_params, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, **data_loader_params)

    # 初始化模型，准备训练
    model = create_mask_model(class_names)
    model = model.to(device, dtype=dtype)
    model.device = device
    model.name = 'maskrcnn_resnet50_fpn_v2'

    # 创建以日期为名称的训练文件夹，保存训练的相关信息
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_dir = Path(project_dir / f"{timestamp}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{model.name}.pth"

    # 设置学习率和训练轮数
    lr = 5e-4
    epochs = 300
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                       max_lr=lr,
                                                       total_steps=epochs * len(train_dataloader))
    train_loop(model=model,
               train_dataloader=train_dataloader,
               valid_dataloader=valid_dataloader,
               optimizer=optimizer,
               lr_scheduler=lr_scheduler,
               device=torch.device(device),
               epochs=epochs,
               checkpoint_path=checkpoint_path,
               use_scaler=True)


def predict(args: argparse):
    """主预测逻辑"""
    model = load_model(args)
    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True)
    ])
    output_path = os.path.join(args.dataset_path, "predictions")
    os.makedirs(output_path, exist_ok=True)

    tracking_data = {}

    for file_name in sorted(os.listdir(args.dataset_path)):  # 按顺序处理帧
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                # 从文件名解析时间戳
                timestamp = parse_timestamp(file_name)
            except ValueError as e:
                print(e)
                continue
            img_path = os.path.join(args.dataset_path, file_name)
            test_img, input_tensor = preprocess_image(img_path, transform)
            pred_bboxes, pred_labels, pred_scores, pred_masks = get_detections(model, input_tensor, args.threshold)
            track_results = update_tracker(pred_bboxes, pred_scores)

            # 更新追踪表
            tracking_table = update_tracking_data(tracking_data, track_results, timestamp)

            # 标注图像
            annotated_img = annotate_image(test_img, track_results, pred_bboxes, pred_masks, FONT_PATH)
            save_image(annotated_img, output_path, file_name)

    with open(os.path.join(output_path, "tracking_table.json"), 'w') as f:
        json.dump(tracking_data, f, indent=4)
    print(f"Tracking table saved to tracking_table.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="命令行参数示例")

    parser.add_argument("--mode", type=str, choices=["train", "predict"], required=False, default='train',
                        help="指定是训练还是预测模式")
    parser.add_argument('--dataset_path', type=str, help="数据集的路径，里面包含图片和标注的json文件", required=True)

    parser.add_argument('--model_path', type=str, help="训练好的模型的路径，如果指定mode=predict，这个参数必须要!")

    parser.add_argument('--threshold', type=float, default=0.7, help="预测阈值，如果指定mode=predict，这个参数必须要!")

    parser.add_argument('--plot', type=bool, default=False, help="绘制json数据,此时dataset_path就是包含json文件的路径")

    args = parser.parse_args()

    # 检查依赖关系
    if args.mode == "predict" and not args.model_path:
        parser.error("--model_path is required when --mode is 'predict'.")

    if args.plot == True and not args.dataset_path:
        parser.error("--dataset_path is required when --plot is 'True'.")

    if args.mode == 'predict':
        predict(args)
    elif args.plot == True:
        plot_tracking_data(args.dataset_path)
    else:
        train(args)
