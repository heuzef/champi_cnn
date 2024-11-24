
# Mushroom Classification Model - JarviSpore

This repository contains **JarviSpore**, a mushroom image classification model trained on a multi-class dataset with 23 different types of mushrooms. Developed from scratch with TensorFlow and Keras, this model aims to provide accurate mushroom identification using advanced deep learning techniques, including *Grad-CAM* for interpreting predictions. This project explores the performance of from-scratch models compared to transfer learning.

## Model Details

- **Architecture**: Custom CNN (Convolutional Neural Network)
- **Number of Classes**: 23 mushroom classes
- **Input Format**: RGB images resized to 224x224 pixels
- **Framework**: TensorFlow & Keras
- **Training**: Conducted on a machine with an i9 14900k processor, 192GB RAM, and an RTX 3090 GPU

## Key Features

1. **Multi-Class Classification**: The model can predict among 23 mushroom species.
2. **Regularization**: Includes L2 regularization and Dropout to prevent overfitting.
3. **Class Weighting**: Manages dataset imbalances by applying specific weights for each class.
4. **Grad-CAM Visualization**: Utilizes Grad-CAM to generate heatmaps, allowing visualization of the regions influencing the model's predictions.

## Model Training

The model was trained using a structured dataset directory with data split as follows:
- `train`: Balanced training dataset
- `validation`: Validation set to monitor performance
- `test`: Test set to evaluate final accuracy

Main training hyperparameters include:
- **Batch Size**: 32
- **Epochs**: 20 with Early Stopping
- **Learning Rate**: 0.0001

Training was tracked and logged via MLflow, including accuracy and loss curves, as well as the best model weights saved automatically.

## Model Usage

### Prerequisites

Ensure the following libraries are installed:
```bash
pip install tensorflow pillow matplotlib numpy
```

### Loading the Model

To load and use the model for predictions:

```python
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the model
model = tf.keras.models.load_model("path_to_model.h5")

# Prepare an image for prediction
def prepare_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Prediction
image_path = "path_to_image.jpg"
img_array = prepare_image(image_path)
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])

print(f"Predicted Class: {predicted_class}")
```

### Grad-CAM Visualization

The integrated *Grad-CAM* functionality allows interpretation of the model's predictions. To use it, select an image and apply the Grad-CAM function to display the heatmap overlaid on the original image, highlighting areas influencing the model.

Grad-CAM example usage:

```python
# Example usage of the make_gradcam_heatmap function
heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name="last_conv_layer_name")

# Superimpose the heatmap on the original image
superimposed_img = superimpose_heatmap(Image.open(image_path), heatmap)
superimposed_img.show()
```

## Evaluation

The model was evaluated on the test set with an average accuracy above random chance, showing promising results for a first from-scratch version.

## Contributing

Contributions to improve accuracy or add new features (e.g., other visualization techniques or advanced optimization) are welcome. Please submit a pull request with relevant modifications.

## License

This model is licensed under a controlled license: please refer to the `LICENSE` file for details. You may use this model for personal projects, but any modifications or redistribution must be approved.

