import numpy as np

# Key layer names for loading weights and managing the YOLOv3 architecture.
YOLOV3_LAYER_LIST = [
    'yolo_darknet',       # Darknet feature extraction backbone
    'yolo_conv_0',        # Convolutional layers for detection head 0
    'yolo_output_0',      # Output layer for detection head 0
    'yolo_conv_1',        # Convolutional layers for detection head 1
    'yolo_output_1',      # Output layer for detection head 1
    'yolo_conv_2',        # Convolutional layers for detection head 2
    'yolo_output_2',      # Output layer for detection head 2
]

# Predefined bounding box sizes, normalized for three scales to detect small, medium, and large objects.
yolo_anchors = np.array([
    (10, 13), (16, 30), (33, 23),   # Small-scale anchor boxes
    (30, 61), (62, 45), (59, 119),  # Medium-scale anchor boxes
    (116, 90), (156, 198), (373, 326)  # Large-scale anchor boxes
], np.float32) / 416  # Normalize by dividing by input size (416)


# Groups of anchors for each detection scale, helping match objects of different sizes.
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])  # Masks for different scales

class_names = [
    'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
    'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]
