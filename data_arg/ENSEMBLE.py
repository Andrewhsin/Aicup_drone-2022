"""
WBF_Ensemble on inference output (*.txt) (inference output must with conference_score)

Weighted boxes fusion
https://github.com/ZFTurbo/Weighted-Boxes-Fusion
"""

import csv
import glob
import cv2
from ensemble_boxes import *
from pathlib import Path

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

# glob (*.txt) from paths and return zip function
def glob_and_zip(*paths, file_type='*.txt'):
    output_list = []
    for path in paths:
        output_list.append(glob.glob(path + file_type))
        # print(output_list)
    return zip(*output_list)

# image dir & list
image_dir = r'/AICUP_aerial_ensemble/public_private/'
image_list = glob.glob(image_dir + '*.png')

# save dir
save_csv_dir = r'./'

name = f'ensemble'

# load txts from model dir path and zip together
# r'D:\AICUP_aerial_ensemble\models\20221006_v7_unknown_8000\exp01_unknown_0.3_0.25_8000'
models2ensemble = [r'./exp01_unknown_0.3_0.25_8000/', r'./exp02_unknown_0.29_0.26_8000/']
zip_txt_list = glob_and_zip(*models2ensemble)

# WBF params
weights = [1] * len(models2ensemble)
iou_thr = 0.4  # 0.3, 0.4, 0.5
skip_box_thr = 0.0001


output_csv_lines_has_conf = []
output_csv_lines_no_conf = []

# WBF_bbox
for index, (image_path, txt_paths) in enumerate(zip(image_list, zip_txt_list)):
    # print(index)
    # continue

    # print(image_path, txt_paths)
    image = cv2.imread(image_path)
    height, width, channels = image.shape

    bboxes_list = []
    scores_list = []
    labels_list = []

    for txt_path in txt_paths:

        img_name = Path(txt_path).stem

        bboxes = []
        scores = []
        labels = []

        with open(txt_path, "r") as txt_file:

            spamreader = csv.reader(txt_file)

            for line in spamreader:
                # print(line)

                # separate `label, x_center, y_center, yolo_w, yolo_h, score(confidence)` from string
                label, x_center, y_center, yolo_w, yolo_h, score = line[0].split()

                # x_min, y_min, x_max, y_max
                bbox = yolo2WBF(x_center, y_center, yolo_w, yolo_h)

                label = int(label)
                score = float(score)

                bboxes.append(bbox)
                scores.append(score)
                labels.append(label)

        bboxes_list.append(bboxes)
        scores_list.append(scores)
        labels_list.append(labels)

    # wbf_bboxes, wbf_scores, wbf_labels
    wbf_bboxes, wbf_scores, wbf_labels = weighted_boxes_fusion(bboxes_list, scores_list, labels_list, weights=weights,
                                                  iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    # print(wbf_bboxes, wbf_scores, wbf_labels)

    for (wbf_label, wbf_bbox, wbf_score) in zip(wbf_labels, wbf_bboxes, wbf_scores):

        coco_bbox = WBF2coco(*wbf_bbox, width, height)
        # print(coco_bbox)
        wbf_label = int(wbf_label)

        output_csv_line_has_conf = [img_name, wbf_label, *coco_bbox, wbf_score]
        output_csv_line_no_conf = [img_name, wbf_label, *coco_bbox]

        output_csv_lines_has_conf.append(output_csv_line_has_conf)
        output_csv_lines_no_conf.append(output_csv_line_no_conf)


# save csv
# csv_has_conf
with open(f'{save_csv_dir}{name}_ensemble_result_{iou_thr}_has_conf.csv', 'w', newline='') as csv_file:

    spamwriter = csv.writer(csv_file)

    for line in output_csv_lines_has_conf:
        spamwriter.writerow(line)

# csv_no_conf
with open(f'{save_csv_dir}{name}_ensemble_result_{iou_thr}_no_conf.csv', 'w', newline='') as csv_file:

    spamwriter = csv.writer(csv_file)

    for line in output_csv_lines_no_conf:
        spamwriter.writerow(line)
