from bambooData import *
from utils import *
import numpy as np
from sort import Sort
import os
import cv2

dataset_path = r"F:\C\Graduation_design\Code\mask-rcnn\cnn\Datasets\origin_data\train\data2"
img_dict, annotation_df = get_Img_Ann(dataset_path, os.path.join(dataset_path, "_annotations.coco.json"))
annotation_df = annotation_df.sort_values(by="image_name")
device = get_torch_device()
tracker = Sort()
FONT_PATH = r"C:\Windows\Fonts\simhei.ttf"  # 替换为你的字体文件路径
transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True)
])

tracking_data = {}


def preprocess_image(img_path, transform):
    """加载并预处理图像"""
    test_img = Image.open(img_path).convert('RGB')
    input_tensor = transform(test_img)[None].to(device)
    return test_img, input_tensor


def update_tracker(pred_bboxes, pred_scores):
    """更新 SORT 追踪器并返回追踪结果"""
    # 转换为 SORT 需要的格式：[x1, y1, x2, y2, score]
    sort_detections = np.array([
        [box[0], box[1], box[2], box[3], score]
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
        mask_idx = np.argmin(np.linalg.norm(np.array(pred_bboxes)[:, :4] - np.array([x1, y1, x2, y2]), axis=1))
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
            # boxes=torch.tensor([[x1, y1, x2, y2]]),
            boxes=pred_bboxes[mask_idx][None],
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

flag=True

if __name__ == '__main__':
    output_path = os.path.join(dataset_path, "predictions")
    os.makedirs(output_path, exist_ok=True)

    for image_name, records in annotation_df.iterrows():
        # 处理data2部分
        if image_name!="P22040811333810_jpg.rf.6e8580aaabe9db1ddd07cac652868954" and flag:
            continue
        else:
            flag=False

        # 处理data1部分
        # if image_name == "P22040811333810_jpg.rf.6e8580aaabe9db1ddd07cac652868954":
        #     break

        shape_points = records['shapes']['points']
        xy_coords = [[tuple(p) for p in points] for points in shape_points]
        # Generate mask images from polygons
        mask_imgs = [create_polygon_mask((1920,1080), xy) for xy in xy_coords]
        # Convert mask images to tensors
        masks = torch.concat([Mask((transforms.PILToTensor()(mask_img)>0).bool(), dtype=torch.bool) for mask_img in mask_imgs])
        bboxes = torchvision.ops.masks_to_boxes(masks)
        label_score = records['shapes']['label']
        image_name += ".jpg"
        timestamp = parse_timestamp(image_name)

        img_path = os.path.join(dataset_path, image_name)
        test_img, input_tensor = preprocess_image(img_path, transform)

        track_results = update_tracker(bboxes, label_score)

        # 更新追踪表
        tracking_table = update_tracking_data(tracking_data, track_results, timestamp,bboxes)

        # 标注图像
        annotated_img = annotate_image(test_img, track_results, bboxes, masks, FONT_PATH)
        save_image(annotated_img, output_path, image_name)
        # break
    with open(os.path.join(output_path, "tracking_table.json"), 'w') as f:
        json.dump(tracking_data, f, indent=4)
    print(f"Tracking table saved to tracking_table.json")
