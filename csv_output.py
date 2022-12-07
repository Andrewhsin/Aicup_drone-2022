import csv
import glob

from pathlib import Path
from PIL import Image

# output(test) image / txt
output_img_dir = 'E:/yolov7-main/runs/detect/exp5/'
output_txt_dir = 'E:/yolov7-main/runs/detect/exp5/labels/'

save_csv_dir = 'E:/yolov7-main/runs/save/'

image_list = glob.glob(output_img_dir + '*.PNG')
txt_list = glob.glob(output_txt_dir + '*.txt')

output_csv_lines = []

for image_index, (img_path, txt_path) in enumerate(zip(image_list, txt_list)):

    im = Image.open(img_path)
    width, height = im.size

    img_name = Path(img_path).stem


    with open(txt_path, "r") as txt_file:

        spamreader = csv.reader(txt_file)

        for line in spamreader:
            """
            classes, x_center, y_center, yolo_w, yolo_h
            to
            img_name, class, x_min, y_min, w, h
            """
            # separate `classes, x_center, y_center, yolo_w, yolo_h` from string
            classes, x_center, y_center, yolo_w, yolo_h = line[0].split()

            # x_min, y_min, w, h
            x_min = round(width * (float(x_center) - float(yolo_w) / 2))
            y_min = round(height * (float(y_center) - float(yolo_h) / 2))
            w = round(width * float(yolo_w))
            h = round(height * float(yolo_h))

            output_csv_line = [img_name, classes, x_min, y_min, w, h]
            output_csv_lines.append(output_csv_line)

    # name -> img0001.txt
    with open(f'{save_csv_dir}result.csv', 'w', newline='') as csv_file:

        spamwriter = csv.writer(csv_file)

        for line in output_csv_lines:
            spamwriter.writerow(line)

