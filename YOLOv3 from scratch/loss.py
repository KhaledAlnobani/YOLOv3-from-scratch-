from tensorflow.keras.losses import (
    binary_crossentropy,               # For computing objectness and confidence losses (binary classification tasks)
    sparse_categorical_crossentropy    # For computing class prediction loss (multi-class classification tasks)
)
import tensorflow as tf
from utils import yolo_boxes, broadcast_iou

def YoloLoss(anchors, classes=80, ignore_thresh=0.5):
    """
    Returns a YOLOv3 loss function.

    Args:
        anchors: anchor boxes for this scale (width, height)
        classes: number of classes in dataset
        ignore_thresh: IoU threshold to ignore predicted boxes

    Returns:
        A function that computes the loss for a batch of predictions.
    """
    def yolo_loss(y_true, y_pred):
        # -------------------------------------------------------
        # Step 1: Transform predicted outputs
        # yolo_boxes decodes raw predictions to bounding boxes in image coordinates
        # pred_xy: center coordinates (x, y)
        # pred_wh: width and height
        # pred_obj: objectness confidence
        # pred_class: class probabilities
        # pred_xywh: xywh format for loss calculation
        # -------------------------------------------------------
        pred_box, pred_obj, pred_class, pred_xywh = yolo_boxes(y_pred, anchors, classes)
        pred_xy = pred_xywh[..., 0:2]  # predicted center x,y
        pred_wh = pred_xywh[..., 2:4]  # predicted width,height
        
        # -------------------------------------------------------
        # Step 2: Transform true outputs
        # Split y_true: true_box(x_min,y_min,x_max,y_max), true_obj, true_class_idx
        # Compute true box centers and widths/heights
        # Box loss scale gives higher weight to small boxes
        # -------------------------------------------------------
        true_box, true_obj, true_class_idx = tf.split(y_true, (4, 1, 1), axis=-1)
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
        true_wh = true_box[..., 2:4] - true_box[..., 0:2]
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

        # -------------------------------------------------------
        # Step 3: Adjust true outputs to match predicted grid format
        # Scale true_xy to grid cell coordinates
        # Take log of true_wh / anchors for width/height prediction
        # Replace infinities (from log(0)) with zeros
        # -------------------------------------------------------
        grid_size = tf.shape(y_true)[1]
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        true_xy = true_xy * tf.cast(grid_size, tf.float32) - tf.cast(grid, tf.float32)
        true_wh = tf.math.log(true_wh / anchors)
        true_wh = tf.where(tf.math.is_inf(true_wh), tf.zeros_like(true_wh), true_wh)

        # -------------------------------------------------------
        # Step 4: Compute masks for object boxes and ignore regions
        # obj_mask: 1 if object exists in this grid cell, else 0
        # ignore_mask: 1 if predicted box IoU < ignore_thresh with any true box
        # -------------------------------------------------------
        obj_mask = tf.squeeze(true_obj, -1)
        true_box_flat = tf.boolean_mask(true_box, tf.cast(obj_mask, tf.bool))  # flatten only boxes with objects
        best_iou = tf.reduce_max(broadcast_iou(pred_box, true_box_flat), axis=-1)
        ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)

        # -------------------------------------------------------
        # Step 5: Calculate individual losses
        # xy_loss: L2 loss on box centers, weighted by obj_mask and box_loss_scale
        # wh_loss: L2 loss on box width/height, weighted similarly
        # obj_loss: binary crossentropy for objectness
        # class_loss: sparse categorical crossentropy for class
        # Only consider object cells for xy, wh, and class loss
        # Ignore cells that have high IoU but no object (via ignore_mask)
        # -------------------------------------------------------
        xy_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        wh_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
        obj_loss = binary_crossentropy(true_obj, pred_obj)
        obj_loss = obj_mask * obj_loss + (1 - obj_mask) * ignore_mask * obj_loss
        class_loss = obj_mask * sparse_categorical_crossentropy(true_class_idx, pred_class)

        # -------------------------------------------------------
        # Step 6: Sum up the losses over all grid cells and anchors
        # axis=(1,2,3) sums over grid_y, grid_x, and anchors
        # Returns a single loss per image
        # -------------------------------------------------------
        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

        # -------------------------------------------------------
        # Step 7: Total loss = sum of all components
        # -------------------------------------------------------
        return xy_loss + wh_loss + obj_loss + class_loss

    return yolo_loss
