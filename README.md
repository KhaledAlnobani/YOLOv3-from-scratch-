# YOLOv3-from-scratch

This project is a TensorFlow implementation of YOLOv3 (You Only Look Once, version 3) — a state-of-the-art real-time object detection model.
It detects multiple objects in images by predicting bounding boxes and class probabilities directly from full images in a single evaluation.

The model was implemented from scratch using TensorFlow and Keras, following the YOLOv3 architecture and anchor-based detection mechanism.
It includes utilities for preprocessing, model construction, target transformation, and result visualization.

# ✨ Key Features

✅ Full YOLOv3 architecture built using TensorFlow/Keras

📦 Support for loading Darknet pre-trained weights

🧩 Image preprocessing and grid-cell target transformation functions

🎯 Bounding box visualization with confidence scores and class names

📊 Modular structure: model.py, loss.py, utils.py, config.py, etc.

# 🧩 Technologies Used

TensorFlow / Keras for model design and training

OpenCV for image processing and drawing bounding boxes

Matplotlib for result visualization

NumPy for numerical operations

# 📚 Learning Resources

This implementation was developed as part of my deep learning journey, referencing:

[Object Detection by YOLO using TensorFlow – GeeksforGeeks] (https://www.geeksforgeeks.org/computer-vision/object-detection-by-yolo-using-tensorflow/)

[Convolutional Neural Networks – Coursera (Deep Learning Specialization)] (https://www.coursera.org/learn/convolutional-neural-networks)
