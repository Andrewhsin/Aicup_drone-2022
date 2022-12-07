import utils.autoanchor as autoAC

new_anchors = autoAC.kmean_anchors('E:/yolov7-main/data/aicup_psu.yaml', 9, 2048, 4.0, 20000, True)
print(new_anchors)