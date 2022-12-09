"""
Data Augmentation by rotate, flip, and more...
"""
from __future__ import annotations
import albumentations as A

import cv2
import glob
from pathlib import Path
import csv

# params
train_angle = 20  # train 1000 (degree:20)[18 * 2] 36x 36000
public_angle = 30  # public 500 24(degrees:30)[12 * 2] 24x 12000


image_dir = 'D:/AICUP_aerial/train_v2_test/'  # trainv2
txt_dir = r'yolo_txt_train_v2_test_psuedo_label_new/'  # (yolo format)

# save img_dir
save_dir = r'/yolo_augmentation/'

# save txt_dir
save_dir_original = f'{save_dir}/txt/'

# check dir exist, if not mkdir
temp = Path(save_dir_original)
temp.mkdir(exist_ok=True)

image_list = glob.glob(image_dir + '*.PNG')
txt_list = glob.glob(txt_dir + '*.txt')

# rotate, flip(h)
def augmentation_methods(degrees: list = [0, 45, 90, 135, 225, 270, 315], format: str = 'yolo', label_fields=['class_labels']):

    transforms = []

    for degree in degrees:

        rot = A.Compose([A.Affine(rotate=degree, p=1, fit_output=True)], bbox_params=A.BboxParams(format=format, label_fields=label_fields))
        flip = A.Compose([A.HorizontalFlip(p=1), A.Affine(rotate=degree, p=1, fit_output=True)], bbox_params=A.BboxParams(format=format, label_fields=label_fields))

        transforms.extend([rot, flip])

    return transforms

train_degrees = [i for i in range(0, 360, train_angle)]
public_degrees = [i for i in range(0, 360, public_angle)]


train_transforms = augmentation_methods(degrees=train_degrees)
public_transforms = augmentation_methods(degrees=public_degrees)

for image_index, (image_path, txt_path) in enumerate(zip(image_list, txt_list)):

    image = cv2.imread(image_path)
    height, width, channels = image.shape

    bboxes = []
    class_labels = []

    with open(txt_path, "r") as txtfile:

            spamreader = csv.reader(txtfile)

            for line in spamreader:

                label, x_center, y_center, yolo_w, yolo_h = line

                bbox = float(x_center), float(y_center), float(yolo_w), float(yolo_h)
                label = int(label)

                bboxes.append(bbox)
                class_labels.append(label)

    if image_index < 1000: #train_transform
        for augment_index, transform in enumerate(train_transforms):

            transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)

            transformed_image = transformed['image']
            transformed_bboxes = transformed['bboxes']
            transformed_class_labels = transformed['class_labels']

            name = f'img{0 + (image_index - 0) * len(train_transforms) + (augment_index + 1):04d}' ### 05d

            print(f'image: {Path(image_path).stem}, its augment: {name}')

            # save image
            cv2.imwrite(f'{save_dir}{name}.png', transformed_image)

            # save txt
            with open(f'{save_dir_original}{name}.txt', 'w') as f_orig:
                for labels, bbox in zip(transformed_class_labels, transformed_bboxes):
                    f_orig.write(f"{labels} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
    else:
        for augment_index, transform in enumerate(public_transforms):

            transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)

            transformed_image = transformed['image']
            transformed_bboxes = transformed['bboxes']
            transformed_class_labels = transformed['class_labels']
            # 1000 * 36 +
            name = f'img{1000 * len(train_transforms)  + (image_index - 1000) * len(public_transforms) + (augment_index + 1):04d}'  ### 05d

            print(f'image: {Path(image_path).stem}, its augment: {name}')

            # save image
            cv2.imwrite(f'{save_dir}{name}.png', transformed_image)

            # save txt
            with open(f'{save_dir_original}{name}.txt', 'w') as f_orig:
                for labels, bbox in zip(transformed_class_labels, transformed_bboxes):
                    f_orig.write(f"{labels} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")