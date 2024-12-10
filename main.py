import datetime
import argparse

from bambooData import *
from utils import *
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

torchvision.disable_beta_transforms_warning()

device = get_torch_device()
dtype = torch.float32
class_names = ['background', 'bamboo']


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
    model = create_mask_model()

    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个 GPU!")
        model = nn.DataParallel(model, dtype=dtype)  # 自动使用所有可用的 GPU
    else:
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
    epochs = 100
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



def predict():
    dataset_path = args.dataset_path
    annotation_file_paths = find_single_json_file(dataset_path)
    img_dict, annotation_df = get_Img_Ann(dataset_path, annotation_file_paths)
    dataset, _ = getDatasets(img_dict, annotation_df, rate=1)

    img_keys = list(img_dict.keys())
    file_id = 'P21033110123310_jpg.rf.a35a6885d20ed2a67529b6cb816650dd'
    test_file = img_dict[file_id]
    test_img = Image.open(test_file).convert('RGB')

    # 获取标注的mask
    annotation = annotation_df.loc[file_id]
    shape_points = annotation['shapes']['points']
    xy_coords = [[tuple(p) for p in points] for points in shape_points]
    mask_imgs = [create_polygon_mask(test_img.size, xy) for xy in xy_coords]
    target_masks = Mask(
        torch.concat([Mask(transforms.PILToTensor()(mask_img), dtype=torch.bool) for mask_img in mask_imgs]))

    # 获取标注的labels
    target_labels = [class_names[i] for i in annotation['shapes']['label']]

    # 获取标注的bboxs
    target_bboxes = BoundingBoxes(data=torchvision.ops.masks_to_boxes(target_masks), format='xyxy',
                                  canvas_size=test_img.size[::-1])

    model = create_mask_model(class_names)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    model.to(device)
    input_tensor = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)])(test_img)[
        None].to(device)
    with torch.no_grad():
        model_output = model(input_tensor)

    threshold = 0.5
    model_output = move_data_to_device(model_output, 'cpu')
    scores_mask = model_output[0]['scores'] > threshold
    pred_bboxes = BoundingBoxes(model_output[0]['boxes'][scores_mask], format='xyxy',
                                canvas_size=test_img.size[::-1])

    # Get the class names for the predicted label indices
    pred_labels = [class_names[int(label)] for label in model_output[0]['labels'][scores_mask]]

    # Extract the confidence scores
    pred_scores = model_output[0]['scores']

    # Scale and stack the predicted segmentation masks
    pred_masks = F.interpolate(model_output[0]['masks'][scores_mask], size=test_img.size[::-1])
    pred_masks = torch.concat([Mask(torch.where(mask >= threshold, 1, 0), dtype=torch.bool) for mask in pred_masks])

    img_tensor = transforms.PILToTensor()(test_img)

    # Annotate the test image with the target segmentation masks
    annotated_tensor = draw_segmentation_masks(image=img_tensor, masks=target_masks, alpha=0.3)
    # Annotate the test image with the target bounding boxes
    annotated_tensor = draw_bboxes(image=annotated_tensor, boxes=target_bboxes, labels=target_labels)
    # Display the annotated test image
    annotated_test_img = tensor_to_pil(annotated_tensor)

    # Annotate the test image with the predicted segmentation masks
    annotated_tensor = draw_segmentation_masks(image=img_tensor, masks=pred_masks, alpha=0.3)
    # Annotate the test image with the predicted labels and bounding boxes
    annotated_tensor = draw_bboxes(
        image=annotated_tensor,
        boxes=pred_bboxes,
        labels=[f"{label}\n{prob * 100:.2f}%" for label, prob in zip(pred_labels, pred_scores)]
    )

    res = stack_imgs([annotated_test_img, tensor_to_pil(annotated_tensor)])
    res.show()
    res.save("res.jpg")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="命令行参数示例")

    parser.add_argument("--mode", type=str, choices=["train", "predict"], required=False, default='train',
                        help="指定是训练还是预测模式")
    parser.add_argument('--dataset_path', type=str, help="数据集的路径，里面包含图片和标注的json文件", required=True)

    parser.add_argument('--model_path',type=str,help="训练好的模型的路径，如果指定mode=predict，这个参数必须要!")

    args = parser.parse_args()

    # 检查依赖关系
    if args.mode == "predict" and not args.model_path:
        parser.error("--model_path is required when --mode is 'predict'.")

    if args.mode == 'predict':
        predict(args)
    else:
        train(args)
