# Model Description and Details

This project includes four models: two Convolutional Neural Network (CNN) models and two YOLO models for object detection. Below are detailed descriptions for each of the models and their configurations.

## 1. CNN Models

### CNN Model 1 (cnn1)
- **Input Shape**: `(222, 222, 3)`
  - **Architecture**:
  - Convolutional layers with filters: 32, 64, 128, 256
  - MaxPooling layers: `(2, 2)`
  - Batch Normalization after each convolutional layer
  - Dropout rate: `0.5` for regularization
  - Fully connected dense layers: 512 neurons
  - Activation function: ReLU for intermediate layers, softmax for the output layer
  - Global Average Pooling after final convolutional block
- **Output**: 15 classes
- **Training Epochs**: 25
- **Key Features**:
  - Designed for high-resolution images with enhanced regularization.
  - Suitable for balanced and detailed feature extraction.

### CNN Model 2 (cnn2)
- **Input Shape**: `(414, 414, 3)`
- **Architecture**:
  - Convolutional layers with filters: 32, 64, 128
  - MaxPooling layers: `(2, 2)`
  - Fully connected dense layers: 256 neurons
  - Dropout rate: `0.5` for regularization
  - Activation function: ReLU for intermediate layers, softmax for the output layer
- **Output**: 15 classes
- **Training Epochs**: 10
- **Key Features**:
  - Trained without oversampling, preserving the natural class distribution.
  - Useful for understanding the model's performance on unbalanced datasets.

## 2. YOLO Models

### YOLO Model 1 (yolo1)
- **Training Epochs**: 10
- **Key Features**:
  - Faster training and inference time due to fewer epochs.
  - Suitable for quick prototyping and applications where speed is critical.

### YOLO Model 2 (yolo2)
- **Training Epochs**: 20
- **Key Features**:
  - Trained with more epochs to achieve higher accuracy.
  - Ideal for scenarios requiring better precision and recall.

## Model Usage
- Both CNN and YOLO models can be used for classification and detection tasks respectively.
- Ensure that the input size matches the expected dimensions during preprocessing to avoid errors.
- Each model is trained for specific tasks and resolutions, and you should select the model based on your use case:
  - Use **cnn1** for high-resolution and regularized data.
  - Use **cnn2** for datasets with natural class distribution.
  - Use **yolo1** for faster detections with less computational overhead.
  - Use **yolo2** for higher detection accuracy.

---

## Model Input and Preprocessing
- **CNNs**:
  - Input images must be resized to the respective model’s input shape before being passed to the model.
  - Normalize pixel values to the range `[0, 1]` by dividing by 255.
- **YOLOs**:
  - YOLO models accept varied input sizes but require consistent resizing during training and inference.

---

## Summary
| Model   | Type | Input Size         | Epochs | Output Classes | Use Case                      |
|---------|------|--------------------|--------|----------------|-------------------------------|
| cnn1    | CNN  | `(222, 222, 3)`   | 25      | 15             | High-resolution, regularized  |
| cnn2    | CNN  | `(414, 414, 3)`   | 10      | 15             | Natural class distribution    |
| yolo1   | YOLO | Variable (Resized) | 10     | Object Detection | Fast prototyping              |
| yolo2   | YOLO | Variable (Resized) | 20     | Object Detection | High accuracy detection       |

