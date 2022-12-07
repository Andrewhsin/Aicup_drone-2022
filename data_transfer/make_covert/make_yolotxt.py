import os
from os import listdir, getcwd
from os.path import join

#E:/yolov7-main/aicup_psu/val/val.txt
#E:/yolov7-main/aicup_psu/val/images/

if __name__ == '__main__':
    source_folder =r'E:/yolov7-main/aicup_psu/val/images/'
    dest = r'E:/yolov7-main/aicup_psu/val/val.txt'
    dest2 = r'E:/yolov7-main/aicup_psu/val/val.txt'
    file_list = os.listdir(source_folder)
    train_file = open(dest, 'a')
    val_file = open(dest2, 'a')
    i=0
    for file_obj in file_list:
        file_name, file_extend = os.path.splitext(file_obj)

        if (i%4 ==0):
           #val_file.write(file_name+".jpg" + '\n')  相對路徑
            val_file.write("E:/yolov7-main/aicup_psu/val/images/"+file_name+".jpg" + '\n')
        else:
           #train_file.write(file_name+".jpg" + '\n')
            train_file.write("E:/yolov7-main/aicup_psu/val/images/"+file_name+".jpg" + '\n')
        i+=1
    train_file.close()
val_file.close()

#E:/yolov7-main/kitti/val/images/
#E:/yolov7-main/kitti/val/val.txt