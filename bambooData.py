import pandas
import torch
from torch.utils.data import Dataset
import torchvision

torchvision.disable_beta_transforms_warning()
from torchvision.tv_tensors import BoundingBoxes, Mask
import random
import json
import pandas as pd
from cjm_pil_utils.core import get_img_files
from PIL import Image, ImageDraw
import torchvision.transforms.v2 as transforms


def create_polygon_mask(image_size, vertices):
    """
    Create a grayscale image with a white polygonal area on a black background.

    Parameters:
    - image_size (tuple): A tuple representing the dimensions (width, height) of the image.
    - vertices (list): A list of tuples, each containing the x, y coordinates of a vertex
                        of the polygon. Vertices should be in clockwise or counter-clockwise order.

    Returns:
    - PIL.Image.Image: A PIL Image object containing the polygonal mask.
    """

    # Create a new black image with the given dimensions
    mask_img = Image.new('L', image_size, 0)

    # Draw the polygon on the image. The area inside the polygon will be white (255).
    ImageDraw.Draw(mask_img, 'L').polygon(vertices, fill=(255))

    # Return the image with the drawn polygon
    return mask_img


def get_Img_Ann(dataset_path: str, annotation_file_paths: str):
    """
    用于得到数据集的字典，解析mask的json文件，便于构建datasets
    """
    img_file_paths = get_img_files(dataset_path)
    img_dict = {file.stem: file for file in img_file_paths}
    # 读取 JSON 文件
    with open(annotation_file_paths, "r") as f:
        data = json.load(f)
    # 初始化列表以存储表格数据
    records = []
    # 遍历 annotations 数据
    count = 0
    annotation = data["annotations"]
    for image_info in data["images"]:
        image_id = image_info['id']
        shapes = []
        labels = []
        while (count < len(annotation) and annotation[count]['image_id'] == image_id):
            coords = annotation[count]['segmentation'][0]
            # 两两一个组成一个xy坐标
            xy = [[coords[i], coords[i + 1]] for i in range(0, len(coords), 2)]
            shapes.append(xy)
            labels.append(annotation[count]["category_id"])
            count += 1

        record = {
            "image_name": image_info["file_name"][:-4],
            "shapes": {
                "points": shapes,
                "label": labels
            }
        }
        records.append(record)

    # 转换为 Pandas DataFrame
    annotation_df = pd.DataFrame(records)
    annotation_df.set_index('image_name', inplace=True)

    return img_dict, annotation_df


def getDatasets(img_dict: dict, ann_df: pandas.DataFrame, rate: float = 0.8):
    img_keys = list(img_dict.keys())
    random.shuffle(img_keys)
    train_split = int(len(img_keys) * rate)

    train_keys = img_keys[:train_split]
    val_keys = img_keys[train_split:]

    # 图像数据集预处理操作
    final_tfms = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),  # 如果 scale=True，则将像素值缩放到 [0, 1] 范围,即归一化操作，很重要！
        transforms.SanitizeBoundingBoxes(),
    ])
    train_dataset = BambooDataset(train_keys, ann_df, img_dict, transforms=final_tfms)
    valid_dataset = BambooDataset(val_keys, ann_df, img_dict, transforms=final_tfms)

    return train_dataset, valid_dataset


class BambooDataset(Dataset):
    """
    This class represents a PyTorch Dataset for a collection of images and their annotations.
    The class is designed to load images along with their corresponding segmentation masks, bounding box annotations, and labels.
    """

    def __init__(self, img_keys, annotation_df, img_dict, transforms=None):
        """
        Constructor for the HagridDataset class.

        Parameters:
        img_keys (list): List of unique identifiers for images.
        annotation_df (DataFrame): DataFrame containing the image annotations.
        img_dict (dict): Dictionary mapping image identifiers to image file paths.
        class_to_idx (dict): Dictionary mapping class labels to indices.
        transforms (callable, optional): Optional transform to be applied on a sample.
        """
        super(Dataset, self).__init__()

        self._img_keys = img_keys  # List of image keys
        self._annotation_df = annotation_df  # DataFrame containing annotations
        self._img_dict = img_dict  # Dictionary mapping image keys to image paths
        self._transforms = transforms  # Image transforms to be applied

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
        int: The number of items in the dataset.
        """
        return len(self._img_keys)

    def __getitem__(self, index):
        """
        Fetch an item from the dataset at the specified index.

        Parameters:
        index (int): Index of the item to fetch from the dataset.

        Returns:
        tuple: A tuple containing the image and its associated target (annotations).
        """
        # Retrieve the key for the image at the specified index
        img_key = self._img_keys[index]
        # Get the annotations for this image
        annotation = self._annotation_df.loc[img_key]
        # Load the image and its target (segmentation masks, bounding boxes and labels)
        image, target = self._load_image_and_target(annotation)

        # Apply the transformations, if any
        if self._transforms:
            image, target = self._transforms(image, target)

        return image, target

    def _load_image_and_target(self, annotation):
        """
        Load an image and its target (bounding boxes and labels).

        Parameters:
        annotation (pandas.Series): The annotations for an image.

        Returns:
        tuple: A tuple containing the image and a dictionary with 'boxes' and 'labels' keys.
        """
        # Retrieve the file path of the image
        filepath = self._img_dict[annotation.name]
        # Open the image file and convert it to RGB
        image = Image.open(filepath).convert('RGB')

        # Convert the class labels to indices
        labels = annotation['shapes']['label']
        labels = torch.Tensor(labels)
        labels = labels.to(dtype=torch.int64)

        # Convert polygons to mask images
        shape_points = annotation['shapes']['points']
        xy_coords = [[tuple(p) for p in points] for points in shape_points]
        mask_imgs = [create_polygon_mask(image.size, xy) for xy in xy_coords]
        masks = Mask(
            torch.concat([Mask(transforms.PILToTensor()(mask_img), dtype=torch.bool) for mask_img in mask_imgs]))

        # Generate bounding box annotations from segmentation masks
        bboxes = BoundingBoxes(data=torchvision.ops.masks_to_boxes(masks), format='xyxy', canvas_size=image.size[::-1])

        return image, {'masks': masks, 'boxes': bboxes, 'labels': labels}
