
import tensorflow as tf
from tensorflow.keras import Model     # Base class for defining YOLOv3 as a custom Keras Model
from tensorflow.keras.regularizers import l2  # L2 regularization to prevent overfitting

from tensorflow.keras.layers import (
    Add,                               # Element-wise addition of feature maps (used in residual connections)
    Concatenate,                       # Concatenates tensors (used when merging feature maps from different layers)
    Conv2D,                            # 2D convolution layer — main building block of CNNs
    Input,                             # Defines input tensors for the model
    Lambda,                            # Wraps arbitrary TensorFlow operations as Keras layers
    LeakyReLU,                         # Activation function — used in YOLO instead of ReLU to allow small negative gradients
    MaxPool2D,                         # Downsampling operation — reduces spatial dimensions while retaining key features
    UpSampling2D,                      # Used to increase spatial dimensions (common in feature pyramid networks)
    ZeroPadding2D                      # Pads images/tensors with zeros to maintain alignment for convolutions
)
from config import yolo_anchors, yolo_anchor_masks

from utils import yolo_nms  # Non-Maximum Suppression to filter overlapping boxes



class BatchNormalization(tf.keras.layers.BatchNormalization):

    """
    Custom BatchNormalization layer that handles the 'training' flag carefully.
    Ensures that frozen layers (trainable=False) do not update moving statistics
    during training.
    """

    @tf.function
    def call(self, x, training=False):
        """
        Applies batch normalization to the input tensor `x`.

        Args:
            x: Input tensor to normalize.
            training: Boolean, True if layer should behave in training mode,
                      False for inference. If None, defaults to False.

        Returns:
            Normalized tensor, using batch statistics if training=True
            and moving statistics if training=False or layer is frozen.
        """

        # If training is None, default to False (inference mode)
        if training is None:
            training = tf.constant(False)

        # Ensure batch norm only updates stats if the layer is trainable
        training = tf.logical_and(training, self.trainable)

        # Call the original Keras BatchNormalization with the modified training flag
        # super() refers to the parent class (tf.keras.layers.BatchNormalization)
        return super(BatchNormalization, self).call(x, training=training)

def DarknetConv(x, filters, size, strides=1, batch_norm=True):
    if strides == 1:
        padding = 'same'
    else:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
        padding = 'valid'
    x = Conv2D(filters=filters, kernel_size=size,
               strides=strides, padding=padding,
               use_bias=not batch_norm, kernel_regularizer=l2(0.0005))(x)
    if batch_norm:
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
    return x

def DarknetResidual(x, filters):
    prev = x
    x = DarknetConv(x, filters // 2, 1)
    x = DarknetConv(x, filters, 3)
    x = Add()([prev, x])  # Ensure addition is valid
    return x  # Return the tensor

def DarknetBlock(x, filters, blocks):
    x = DarknetConv(x, filters, 3, strides=2)
    for _ in range(blocks):
        x = DarknetResidual(x, filters)  # Ensure residual connection
    return x  # Return the tensor

def Darknet(name=None):
    x = inputs = Input([None, None, 3])
    x = DarknetConv(x, 32, 3)
    x = DarknetBlock(x, 64, 1)
    x = DarknetBlock(x, 128, 2)  # skip connection
    x = x_36 = DarknetBlock(x, 256, 8)  # skip connection
    x = x_61 = DarknetBlock(x, 512, 8)
    x = DarknetBlock(x, 1024, 4)
    return tf.keras.Model(inputs, (x_36, x_61, x), name=name)

def YoloConv(x_in, filters, name=None):
    if isinstance(x_in, tuple):  # Skip connection
        inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
        x, x_skip = inputs
        x = DarknetConv(x, filters, 1)
        x = UpSampling2D(2)(x)
        x = Concatenate()([x, x_skip])
    else:
        x = inputs = Input(x_in.shape[1:])
    x = DarknetConv(x, filters, 1)
    x = DarknetConv(x, filters * 2, 3)
    x = DarknetConv(x, filters, 1)
    x = DarknetConv(x, filters * 2, 3)
    x = DarknetConv(x, filters, 1)
    return Model(inputs, x, name=name)(x_in)

def YoloOutput(x_in, filters, anchors, classes, name=None):
    ''' Creates a YOLO output layer for predictions.
    Parameters:
    x_in   : Input tensor from previous layer
    
    filters : Number of filters for the intermediate convolution
    anchors : Number of anchor boxes for this scale
    classes : Number of object classes to predict
    name    : Name for the Keras model
    Returns:
    A Keras model that outputs predictions in the shape:
    (batch, grid_h, grid_w, anchors, classes + 5)
    where 5 = 4 box coords + 1 objectness score
    '''
    # Create a Keras input layer matching the input feature map shape
    x = inputs = Input(x_in.shape[1:])  # shape: (height, width, channels)

    # Apply a 3x3 convolution to increase feature richness
    # Filters = filters*2, stride=1, includes BatchNorm + LeakyReLU
    x = DarknetConv(x, filters * 2, 3)

    # Apply a 1x1 convolution to produce final predictions for this scale
    # Number of filters = anchors * (classes + 5)
    # 4 box coords + 1 objectness + class probabilities
    # batch_norm=False because prediction layer doesn't use BatchNorm
    x = DarknetConv(x, anchors * (classes + 5), 1, batch_norm=False)

    # Reshape output to separate predictions per anchor
    # Resulting shape: (batch, grid_h, grid_w, anchors, classes+5)
    x = Lambda(lambda x: tf.reshape(
        x, (-1, tf.shape(x)[1], tf.shape(x)[2], anchors, classes + 5)
    ))(x)

    # Build a Keras model and immediately call it on x_in to get output tensor
    return tf.keras.Model(inputs, x, name=name)(x_in)


def yolo_boxes(pred, anchors, classes):
    """
    Decodes YOLO model predictions into interpretable bounding boxes.

    Args:
        pred: Tensor of shape (batch, grid, grid, anchors, (x, y, w, h, obj, ...classes))
              → raw model output for one YOLO scale.
        anchors: Tensor/list of anchor box dimensions [(w, h), ...] for this scale.
        classes: Integer → number of object classes.

    Returns:
        bbox: Tensor of shape (batch, grid, grid, anchors, 4)
              → final box coordinates [x1, y1, x2, y2] normalized to image size.
        objectness: Probability each box contains an object.
        class_probs: Class probabilities for each box.
        pred_box: Tensor of [x, y, w, h] (before decoding) used in loss computation.
    """

    # -------------------------------------------------------
    # Get the grid size (e.g., 13 for 13x13 feature map)
    # -------------------------------------------------------
    grid_size = tf.shape(pred)[1]

    # -------------------------------------------------------
    # Split the raw YOLO output into its components:
    #   box_xy  → predicted center (x, y) coordinates (relative to cell)
    #   box_wh  → predicted width and height (log-space)
    #   objectness → confidence that an object exists in this box
    #   class_probs → class probabilities
    # -------------------------------------------------------
    box_xy, box_wh, objectness, class_probs = tf.split(
        pred, (2, 2, 1, classes), axis=-1
    )

    # -------------------------------------------------------
    # Apply sigmoid to normalize values:
    #   - box_xy in [0, 1] within each cell
    #   - objectness in [0, 1]
    #   - class_probs in [0, 1]
    # -------------------------------------------------------
    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)

    # -------------------------------------------------------
    # Combine box_xy and box_wh again for loss calculation
    # (still in the raw predicted space)
    # -------------------------------------------------------
    pred_box = tf.concat((box_xy, box_wh), axis=-1)

    # -------------------------------------------------------
    # Build a coordinate grid representing each cell position
    # -------------------------------------------------------
    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))

    # -------------------------------------------------------
    # Stack grid_x and grid_y → [[[0,0],[1,0],[2,0]], [[0,1],[1,1],[2,1]], ...]
    # Expand dimensions so each cell can have multiple anchors
    # Final shape: [grid_size, grid_size, 1, 2]
    # -------------------------------------------------------
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)

    # -------------------------------------------------------
    # Decode box centers:
    #   - Add grid offset to move from cell coordinates to image coordinates
    #   - Divide by grid_size to normalize (0 → left/top, 1 → right/bottom)
    # -------------------------------------------------------
    box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)

    # -------------------------------------------------------
    # Decode box sizes:
    #   - Apply exp() to convert from log-space
    #   - Multiply by anchor box dimensions to scale properly
    # -------------------------------------------------------
    box_wh = tf.exp(box_wh) * anchors

    # -------------------------------------------------------
    # Convert from center (x, y, w, h) → corners (x1, y1, x2, y2)
    # x1,y1 → top-left corner
    # x2,y2 → bottom-right corner
    # -------------------------------------------------------
    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2

    # -------------------------------------------------------
    # Combine corners into final bounding box coordinates
    # -------------------------------------------------------
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    # -------------------------------------------------------
    # Return:
    #   bbox        → decoded boxes [x1, y1, x2, y2]
    #   objectness  → probability of object presence
    #   class_probs → probabilities of each class
    #   pred_box    → original raw predictions [x, y, w, h]
    # -------------------------------------------------------
    return bbox, objectness, class_probs, pred_box
def YoloV3(size=None, channels=3, anchors=yolo_anchors, masks=yolo_anchor_masks, classes=80, training=False):
    """
    Builds the YOLOv3 model.
    
    Args:
        size: Input image size (height=width=size)
        channels: Number of image channels (default 3 for RGB)
        anchors: Anchor boxes for all scales
        masks: Anchor masks for each scale
        classes: Number of object classes
        training: If True, returns raw outputs for loss computation; if False, returns post-processed boxes
    
    Returns:
        A Keras Model instance for YOLOv3.
    """

    # -------------------------------------------------------
    # Step 1: Define input layer
    # Input shape: (size, size, channels)
    # -------------------------------------------------------
    x = inputs = Input([size, size, channels])

    # -------------------------------------------------------
    # Step 2: Extract features from Darknet backbone
    # x_36, x_61, x are outputs from specific layers for skip connections
    # -------------------------------------------------------
    x_36, x_61, x = Darknet(name='yolo_darknet')(x)

    # -------------------------------------------------------
    # Step 3: YOLO detection head for scale 0 (smallest feature map, largest receptive field)
    # -------------------------------------------------------
    x = YoloConv(x, 512, name='yolo_conv_0')                       # Convolutional layers for detection
    output_0 = YoloOutput(x, 512, len(masks[0]), classes, name='yolo_output_0')  # Output layer for predictions

    # -------------------------------------------------------
    # Step 4: YOLO detection head for scale 1 (medium feature map)
    # Uses skip connection from x_61
    # -------------------------------------------------------
    x = YoloConv((x, x_61), 256, name='yolo_conv_1')
    output_1 = YoloOutput(x, 256, len(masks[1]), classes, name='yolo_output_1')

    # -------------------------------------------------------
    # Step 5: YOLO detection head for scale 2 (largest feature map)
    # Uses skip connection from x_36
    # -------------------------------------------------------
    x = YoloConv((x, x_36), 128, name='yolo_conv_2')
    output_2 = YoloOutput(x, 128, len(masks[2]), classes, name='yolo_output_2')

    # -------------------------------------------------------
    # Step 6: If training=True, return raw outputs for loss computation
    # Each output contains predictions in the format [x, y, w, h, objectness, class probabilities]
    # -------------------------------------------------------
    if training:
        return Model(inputs, (output_0, output_1, output_2), name='yolov3')

    # -------------------------------------------------------
    # Step 7: Convert raw outputs to bounding boxes, objectness, and class probabilities
    # yolo_boxes function decodes the predictions into actual box coordinates
    # -------------------------------------------------------
    boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes), name='yolo_boxes_0')(output_0)
    boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes), name='yolo_boxes_1')(output_1)
    boxes_2 = Lambda(lambda x: yolo_boxes(x, anchors[masks[2]], classes), name='yolo_boxes_2')(output_2)

    # -------------------------------------------------------
    # Step 8: Apply Non-Maximum Suppression to remove duplicate/overlapping boxes
    # yolo_nms function combines boxes from all scales and performs NMS per class
    # boxes_*[:3] slices to get (bbox, objectness, class_probs) for NMS
    # -------------------------------------------------------
    outputs = Lambda(lambda x: yolo_nms(x, anchors, masks, classes), name='yolo_nms')(
        (boxes_0[:3], boxes_1[:3], boxes_2[:3])
    )

    # -------------------------------------------------------
    # Step 9: Create final inference model
    # Input: image
    # Output: final boxes, scores, classes, number of valid detections
    # -------------------------------------------------------
    return Model(inputs, outputs, name='yolov3')




