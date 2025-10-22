# Numerical and data manipulation libraries
import numpy as np                     # For efficient numerical computations, array and matrix operations
import pandas as pd                    # For data handling, analysis, and working with tabular datasets (e.g., annotations, labels)

# Computer vision and file handling
import cv2                             # OpenCV — used for image processing, loading images, resizing, drawing bounding boxes, etc.
import os                              # For interacting with the operating system (e.g., file paths, directory operations)
import glob                            # For searching and matching file paths using patterns (e.g., "*.jpg")

# XML parsing (used for reading Pascal VOC-style annotation files)
import xml.etree.ElementTree as ET     # To parse and extract bounding box info (class, coordinates) from XML annotation files

# Deep learning and model building
import tensorflow as tf                # Core TensorFlow library — provides tensors, GPU acceleration, and deep learning utilities

# Data visualization
import matplotlib.pyplot as plt        # For plotting and displaying images, training curves, and bounding boxes

# Keras model and layers (high-level TensorFlow API)

# Loss functions for training
from tensorflow.keras.regularizers import l2  # L2 regularization to prevent overfitting by penalizing large weights
from config import YOLOV3_LAYER_LIST  # Importing layer names for loading weights

def load_darknet_weights(model, weights_file):
    '''
    Loads pre-trained YOLOv3 weights from a Darknet-format .weights file into a Keras model.
    Parameters:
    model        : Keras model instance (YOLOv3 architecture)
    weights_file : Path to the Darknet .weights file
    '''
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)
    
    layers = YOLOV3_LAYER_LIST  
    
    for layer_name in layers:
        sub_model = model.get_layer(layer_name)
        for i, layer in enumerate(sub_model.layers):
            if not layer.name.startswith('conv2d'):
                continue
            batch_norm = None
            if i + 1 < len(sub_model.layers) and sub_model.layers[i + 1].name.startswith('batch_norm'):
                batch_norm = sub_model.layers[i + 1]
            filters = layer.filters
            size = layer.kernel_size[0]
            in_dim = layer.input.shape[-1]  # Use layer.input.shape for input dimension
            if batch_norm is None:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
            else:
                bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
                
            conv_shape = (filters, in_dim, size, size)
            conv_weights = np.fromfile(wf, dtype=np.float32, count=int(np.prod(conv_shape)))
            conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])
            
            if batch_norm is None:
                layer.set_weights([conv_weights, conv_bias])
            else:
                layer.set_weights([conv_weights])
                batch_norm.set_weights(bn_weights)
    
    assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()

def broadcast_iou(box_1, box_2):
    # ------------------------------
    # Step 1: Prepare for broadcasting
    # ------------------------------
    # Add extra dimensions so TensorFlow can automatically compare
    # every predicted box with every ground-truth box without loops.
    # box_1: (num_preds, 4) -> (num_preds, 1, 4)
    # box_2: (num_truth, 4) -> (1, num_truth, 4)
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)
    
    # ------------------------------
    # Step 2: Broadcast to compatible shapes
    # ------------------------------
    # Determine the shape that both tensors can be broadcasted to
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    # Expand both tensors to the new shape
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)
    # After this:
    # box_1 and box_2 both have shape: (num_preds, num_truth, 4)
    # Each predicted box is now paired with each ground-truth box

    # ------------------------------
    # Step 3: Calculate intersection width and height
    # ------------------------------
    # Find the overlapping width (x-axis) of each box pair
    int_w = tf.maximum(
        tf.minimum(box_1[..., 2], box_2[..., 2]) - tf.maximum(box_1[..., 0], box_2[..., 0]),
        0
    )
    # Find the overlapping height (y-axis) of each box pair
    int_h = tf.maximum(
        tf.minimum(box_1[..., 3], box_2[..., 3]) - tf.maximum(box_1[..., 1], box_2[..., 1]),
        0
    )
    # Intersection area = width * height
    int_area = int_w * int_h

    # ------------------------------
    # Step 4: Calculate area of each box
    # ------------------------------
    # Area of predicted boxes
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * (box_1[..., 3] - box_1[..., 1])
    # Area of ground-truth boxes
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * (box_2[..., 3] - box_2[..., 1])

    # ------------------------------
    # Step 5: Compute IOU
    # ------------------------------
    # IOU = Intersection / Union
    return int_area / (box_1_area + box_2_area - int_area)


def freeze_all(model, frozen=True):
    # Set the trainable property of the model/layer
    # If frozen=True -> trainable=False (weights will NOT be updated during training)
    # If frozen=False -> trainable=True (weights can be updated)
    model.trainable = not frozen

    # If the model has sub-layers (i.e., it's a Keras Model)
    if isinstance(model, tf.keras.Model):
        # Loop through all sub-layers and apply the same freezing recursively
        for l in model.layers:
            freeze_all(l, frozen)  # Recursive call ensures nested layers are also frozen/unfrozen


def draw_outputs(img, outputs, class_names):
    '''
    Draws bounding boxes and class labels on an image.
    
    Parameters:
    img          : The original image (NumPy array)
    outputs      : YOLO model outputs [boxes, objectness, classes, nums]
    class_names  : List of class names corresponding to class IDs
    
    Returns:
    img          : Image with drawn bounding boxes and labels
    '''

    # Unpack the outputs
    boxes, objectness, classes, nums = outputs
    # For batch size = 1, take the first element of each
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    
    # Get image width and height, flipped because boxes are normalized [0,1]
    wh = np.flip(img.shape[0:2])  # wh = (width, height)
    
    # Loop over each detected object
    for i in range(nums):
        # Convert normalized box coordinates to pixel values
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))  # top-left corner
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))  # bottom-right corner
        
        # Draw the bounding box on the image
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)  # Blue box with thickness 2
        
        # Put label: class name and confidence score
        img = cv2.putText(
            img,
            '{} {:.4f}'.format(class_names[int(classes[i])], objectness[i]),  # label text
            x1y1,  # location to draw text (top-left corner of box)
            cv2.FONT_HERSHEY_COMPLEX_SMALL,  # font type
            1,  # font scale
            (0, 0, 255),  # color (red)
            2  # thickness of text
        )
    
    # Return the image with drawn boxes and labels
    return img


def transform_images(x_train, size):
    '''
    Preprocesses an image for model input.
    
    Parameters:
    x_train : input image (or batch of images) as a NumPy array or Tensor
    size    : target size (width and height) to resize the image
    
    Returns:
    x_train : preprocessed image ready for the model
    '''
    
    # Resize the image to (size, size)
    # YOLO expects square images of a fixed size (e.g., 416x416)
    x_train = tf.image.resize(x_train, (size, size))
    
    # Normalize pixel values from [0, 255] to [0, 1]
    # Neural networks usually train better with normalized inputs
    x_train = x_train / 255.0
    
    # Return the processed image
    return x_train

def transform_targets_for_output(y_true, grid_size, anchor_idxs, classes):
    """
    Prepares ground-truth bounding boxes for one YOLO output scale.

    Args:
        y_true: Tensor of shape (batch_size, max_boxes, 6)
                Each box: [x_center, y_center, w, h, class_id, anchor_best_index]
        grid_size: Integer -> number of cells along width/height for this YOLO scale
        anchor_idxs: Tensor/list of anchor indices (e.g., [6,7,8] for scale 13x13)
        anchors: Tensor of all anchors [(w,h), (w,h), ...]
        num_classes: total number of classes

    Returns:
        y_true_out: Tensor of shape
            (batch_size, grid_size, grid_size, num_anchors, 6)
            → each entry: [x, y, w, h, objectness, class_id]
    """

   # -------------------------------------------------------
    # Step 1: Get batch size (number of images in this batch)
    # -------------------------------------------------------
    N = tf.shape(y_true)[0]

    # -------------------------------------------------------
    # Step 2: Initialize the output tensor for this YOLO layer
    # Shape: (batch, grid_size, grid_size, num_anchors_for_this_layer, 6)
    # 6 = [x_min, y_min, x_max, y_max, objectness, class_id]
    # Initially all zeros (no object in grid cells)
    # -------------------------------------------------------
    y_true_out = tf.zeros(
        (N, grid_size, grid_size, tf.shape(anchor_idxs)[0], 6)
    )

    # -------------------------------------------------------
    # Step 3: Make sure anchor indices are integers for comparisons
    # -------------------------------------------------------
    anchor_idxs = tf.cast(anchor_idxs, tf.int32)

    # -------------------------------------------------------
    # Step 4: Initialize TensorArrays to collect indices and values for scatter update
    # We use TensorArrays because we cannot dynamically modify normal tensors in tf.function loops
    # -------------------------------------------------------
    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)   # positions to update
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True) # values to write

    # Counter to keep track of number of updates written
    idx = 0

    # -------------------------------------------------------
    # Step 5: Loop over each image in the batch
    # -------------------------------------------------------
    for i in tf.range(N):
        # Loop over each box in the image
        for j in tf.range(tf.shape(y_true)[1]):
            # Skip boxes with zero width (padded entries)
            if tf.equal(y_true[i][j][2], 0):
                continue

            # -------------------------------------------------------
            # Step 6: Check if the box's anchor matches any anchor in this layer
            # -------------------------------------------------------
            anchor_eq = tf.equal(
                anchor_idxs, tf.cast(y_true[i][j][5], tf.int32)
            )

            # -------------------------------------------------------
            # Step 7: If any anchor matches, process this box
            # tf.reduce_any(anchor_eq) → True if any match found
            # -------------------------------------------------------
            if tf.reduce_any(anchor_eq):
                # Extract box coordinates [x_min, y_min, x_max, y_max]
                box = y_true[i][j][0:4]

                # Compute box center (x_center, y_center)
                box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2

                # Find the index of the matching anchor for this layer
                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)

                # Determine which grid cell the box center belongs to
                grid_xy = tf.cast(box_xy // (1/grid_size), tf.int32)

                # -------------------------------------------------------
                # Step 8: Record the position in the output tensor to update
                # Format: [batch_index, grid_y, grid_x, anchor_index]
                # -------------------------------------------------------
                indexes = indexes.write(
                    idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]]
                )

                # -------------------------------------------------------
                # Step 9: Record the actual box values to write
                # 6 values: [x, y, w, h, objectness=1, class_id]
                # -------------------------------------------------------
                updates = updates.write(
                    idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]]
                )

                # Increment update counter
                idx += 1

    # -------------------------------------------------------
    # Step 10: Apply the collected updates to the output tensor
    # tensor_scatter_nd_update inserts all values at the collected indices
    # indexes.stack() → shape (num_updates, 4)
    # updates.stack() → shape (num_updates, 6)
    # -------------------------------------------------------
    return tf.tensor_scatter_nd_update(
        y_true_out, indexes.stack(), updates.stack()
    )

def transform_targets(y_train, anchors, anchor_masks, classes):
    """
    Prepares the ground-truth boxes for all YOLO output scales (13x13, 26x26, 52x52).
    
    Args:
        y_train: Tensor of shape (batch, max_boxes, 5) 
                 Each box: [x_min, y_min, x_max, y_max, class_id]
        anchors: Tensor of all anchor boxes, shape (num_anchors, 2) -> [w, h]
        anchor_masks: List of anchor index groups per YOLO scale (e.g., [[0,1,2],[3,4,5],[6,7,8]])
        classes: Total number of classes in the dataset
        
    Returns:
        Tuple of transformed targets for each scale, each of shape:
        (batch, grid_size, grid_size, num_anchors_per_scale, 6)
        → 6 = [x, y, w, h, objectness, class_id]
    """
    
    # -------------------------------------------------------
    # Step 1: Initialize output list for each YOLO scale
    # -------------------------------------------------------
    y_outs = []

    # Step 2: Start with the smallest grid size (13x13 for YOLOv3)
    grid_size = 13

    # Step 3: Convert anchors to float32
    anchors = tf.cast(anchors, tf.float32)

    # -------------------------------------------------------
    # Step 4: Compute area of each anchor box
    # anchors[...,0] = width, anchors[...,1] = height
    # -------------------------------------------------------
    anchor_area = anchors[..., 0] * anchors[..., 1]

    # -------------------------------------------------------
    # Step 5: Compute width and height of each ground truth box
    # box_wh = [w, h] = x_max - x_min, y_max - y_min
    # -------------------------------------------------------
    box_wh = y_train[..., 2:4] - y_train[..., 0:2]

    # -------------------------------------------------------
    # Step 6: Expand box_wh to shape (batch, max_boxes, num_anchors, 2)
    # so we can compute IoU with all anchors at once
    # -------------------------------------------------------
    box_wh = tf.tile(tf.expand_dims(box_wh, -2), (1, 1, tf.shape(anchors)[0], 1))

    # -------------------------------------------------------
    # Step 7: Compute area of each ground truth box
    # -------------------------------------------------------
    box_area = box_wh[..., 0] * box_wh[..., 1]

    # -------------------------------------------------------
    # Step 8: Compute intersection area between each box and each anchor
    # min(box_w, anchor_w) * min(box_h, anchor_h)
    # -------------------------------------------------------
    intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * tf.minimum(box_wh[..., 1], anchors[..., 1])

    # -------------------------------------------------------
    # Step 9: Compute IoU between each box and each anchor
    # IoU = intersection / (box_area + anchor_area - intersection)
    # -------------------------------------------------------
    iou = intersection / (box_area + anchor_area - intersection)

    # -------------------------------------------------------
    # Step 10: Find the anchor index with the highest IoU for each box
    # This is the “best anchor” that matches the ground truth box
    # -------------------------------------------------------
    anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
    anchor_idx = tf.expand_dims(anchor_idx, axis=-1)  # shape: (batch, max_boxes, 1)

    # -------------------------------------------------------
    # Step 11: Append the best anchor index to the ground truth boxes
    # New y_train shape: (batch, max_boxes, 6)
    # [x_min, y_min, x_max, y_max, class_id, best_anchor_index]
    # -------------------------------------------------------
    y_train = tf.concat([y_train, anchor_idx], axis=-1)

    # -------------------------------------------------------
    # Step 12: Transform ground truth boxes for each scale (YOLO layers)
    # Uses the transform_targets_for_output function defined earlier
    # -------------------------------------------------------
    for anchor_idxs in anchor_masks:
        # Transform boxes for this scale
        y_outs.append(transform_targets_for_output(
            y_train, grid_size, anchor_idxs, classes
        ))
        # Double grid size for next scale (13 → 26 → 52)
        grid_size *= 2

    # -------------------------------------------------------
    # Step 13: Return a tuple of transformed targets for all scales
    # Each element: (batch, grid_size, grid_size, num_anchors_per_scale, 6)
    # -------------------------------------------------------
    return tuple(y_outs)



def yolo_nms(outputs, anchors, masks, classes):
    """
    Applies Non-Maximum Suppression (NMS) to YOLO predictions across all scales.

    Args:
        outputs: list of tuples (boxes, objectness, class_probs) from each YOLO output scale
        anchors: anchor boxes (not used directly here, kept for compatibility)
        masks: anchor masks for each scale (not used here, kept for compatibility)
        classes: number of classes

    Returns:
        boxes: filtered bounding boxes after NMS, shape (batch, max_total_size, 4)
        scores: objectness*class probability for each box, shape (batch, max_total_size)
        classes: class index for each box, shape (batch, max_total_size)
        valid_detections: number of valid boxes per image, shape (batch,)
    """

    # -------------------------------------------------------
    # Step 1: Initialize empty lists to collect predictions
    # b -> boxes, c -> objectness/confidence, t -> class probabilities
    # -------------------------------------------------------
    b, c, t = [], [], []

    # -------------------------------------------------------
    # Step 2: Loop over each scale output (YOLO has 3 scales)
    # -------------------------------------------------------
    for o in outputs:
        # o[0] -> boxes: (batch, grid, grid, anchors, 4)
        # Flatten grid and anchors into one dimension per image
        # Shape after reshape: (batch, num_boxes_per_scale, 4)
        b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))

        # o[1] -> objectness/confidence: (batch, grid, grid, anchors, 1)
        # Flatten same way: (batch, num_boxes_per_scale, 1)
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))

        # o[2] -> class probabilities: (batch, grid, grid, anchors, num_classes)
        # Flatten same way: (batch, num_boxes_per_scale, num_classes)
        t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

    # -------------------------------------------------------
    # Step 3: Concatenate all scales along the box dimension
    # axis=1 ensures all boxes are merged per image
    # -------------------------------------------------------
    bbox = tf.concat(b, axis=1)          # (batch, total_boxes, 4)
    confidence = tf.concat(c, axis=1)    # (batch, total_boxes, 1)
    class_probs = tf.concat(t, axis=1)   # (batch, total_boxes, num_classes)

    # -------------------------------------------------------
    # Step 4: Compute per-class scores for each box
    # Multiply objectness by class probabilities
    # -------------------------------------------------------
    scores = confidence * class_probs     # (batch, total_boxes, num_classes)

    # -------------------------------------------------------
    # Step 5: Apply Non-Maximum Suppression
    # tf.image.combined_non_max_suppression handles:
    # - per-class NMS
    # - maximum boxes per class and per image
    # - filtering by IoU and score thresholds
    # -------------------------------------------------------
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),  # reshape for NMS
        scores=tf.reshape(
            scores,
            (tf.shape(scores)[0], -1, tf.shape(scores)[-1])      # reshape scores
        ),
        max_output_size_per_class=100,  # max 100 boxes per class
        max_total_size=100,             # max 100 boxes per image
        iou_threshold=0.5,              # IoU threshold to filter overlapping boxes
        score_threshold=0.5             # minimum score to keep a box
    )

    # -------------------------------------------------------
    # Step 6: Return the final filtered boxes, scores, classes
    # and number of valid detections per image
    # -------------------------------------------------------
    return boxes, scores, classes, valid_detections



