"""
Pseudo_label write_txt (from csv)
"""

import csv
import glob
import cv2
from ensemble_boxes import *
from pathlib import Path
import pandas as pd

# helper function
def yolo2WBF(x_center, y_center, yolo_w, yolo_h):

    # norm [0...1]
    norm_x_min = float(x_center) - float(yolo_w) / 2
    norm_y_min = float(y_center) - float(yolo_h) / 2
    norm_x_max = float(x_center) + float(yolo_w) / 2
    norm_y_max = float(y_center) + float(yolo_h) / 2

    # x1, y1, x2, y2
    return [norm_x_min, norm_y_min, norm_x_max, norm_y_max]

def WBF2coco(norm_x_min, norm_y_min, norm_x_max, norm_y_max, width, height):
    x_min = round(int(width) * float(norm_x_min))
    y_min = round(int(height) * float(norm_y_min))
    w = round(int(width) * (float(norm_x_max) - float(norm_x_min)))
    h = round(int(height) * (float(norm_y_max) - float(norm_y_min)))

    # x_min, y_min, w, h
    return [x_min, y_min, w, h]

def yolo2coco(x_center, y_center, yolo_w, yolo_h, width, height):

    x_min = round(int(width) * (float(x_center) - float(yolo_w) / 2))
    y_min = round(int(height) * (float(y_center) - float(yolo_h) / 2))
    w = round(int(width) * float(yolo_w))
    h = round(int(height) * float(yolo_h))

    # x_min, y_min, w, h
    return [x_min, y_min, w, h]

def coco2WBF(x_min, y_min, w, h, width, height):
    norm_x_min = int(x_min) / int(width)
    norm_y_min = int(y_min) / int(height)
    norm_x_max = (int(x_min) + int(w)) / int(width)
    norm_y_max = (int(y_min) + int(h)) / int(height)

    # norm_x_min, norm_y_min, norm_x_max, norm_y_max
    return [norm_x_min, norm_y_min, norm_x_max, norm_y_max]

def coco2yolo(x_min, y_min, w, h, width, height):

    x_center = (int(x_min) + int(w) / 2) / int(width)
    y_center = (int(y_min) + int(h) / 2) / int(height)
    yolo_w = int(w) / int(width)
    yolo_h = int(h) / int(height)

    # x_center, y_center, yolo_w, yolo_h
    return [x_center, y_center, yolo_w, yolo_h]



# load csv
csv_path = r'/exp.csv'

col_names = ['name', 'label', 'x_min', 'y_min', 'w', 'h']
df = pd.read_csv(csv_path, names=col_names, index_col=col_names[0])

# load image & to list
image_dir = f'D:/AICUP_aerial_ensemble/public/'
image_list = glob.glob(image_dir + '*.png')

# save dir
save_dir = r'./pseudo_label/'


for index, image_path in enumerate(image_list):

    # image
    image = cv2.imread(image_path)
    height, width, channels = image.shape
    img_name = Path(image_path).stem

    # csv
    # [] brackets matter, can turn one-column pd.Series to pd.DataFrame
    select_df = df.loc[[img_name]]

    yolo_txt_lines = []

    for index, row in select_df.iterrows():

        # ['label', 'x_min', 'y_min', 'w', 'h', 'conf']
        #         ->|           BBOX          |<-

        label = row['label']  # value
        bbox_coco = row['x_min':'h'].tolist()  # list
        # score = row['conf']  # value

        # coco2yolo
        bbox_yolo = coco2yolo(*bbox_coco, width, height)
        label = int(label)

        line = label, *bbox_yolo
        yolo_txt_lines.append(line)

    # txt
    with open(f'{save_dir}/{img_name}.txt', 'w', newline='') as txt_file:

        for line in yolo_txt_lines:
            txt_file.write(f"{line[0]},{line[1]},{line[2]},{line[3]},{line[4]}\n")




