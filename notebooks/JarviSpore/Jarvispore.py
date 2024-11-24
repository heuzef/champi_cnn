# *****************************
# ***** Cleaned Imports *****
# *****************************

# standard libraries
import os  # interact with the operating system

# third-party libraries for numerical and data processing
import numpy as np  # numeric calculations and array handling
import pandas as pd  # data manipulation and analysis

# machine learning and deep learning libraries
import tensorflow as tf  # ML and DL framework
from tensorflow.keras import layers, models  # build neural network layers and models
from tensorflow.keras.preprocessing import image_dataset_from_directory  # load image datasets from directories
from tensorflow.keras.models import Model  # base model class in Keras
from tensorflow.keras.regularizers import l2  # L2 regularization to prevent overfitting
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # augment image data

# image processing and visualization libraries
import cv2  # image and video processing
from PIL import Image  # handle image files
import matplotlib.pyplot as plt  # create visual data representations

# evaluation and tracking libraries
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay  # model performance evaluation
from sklearn.utils import class_weight  # compute class weights for unbalanced data
from sklearn import __version__ as sklearn_version  # retrieve scikit-learn version

# experiment tracking and HTTP requests
import mlflow  # track ML experiments
import requests  # perform HTTP requests

import tkinter as tk
from tkinter import filedialog
import random


# *****************************
# ***** Extracting Library Versions *****
# *****************************

# define versions dictionary
versions = {
    "os": "N/A",  # os does not have accessible version
    "numpy": np.__version__,
    "tensorflow": tf.__version__,
    "matplotlib": matplotlib.__version__,
    "scikit-learn": sklearn_version,
    "PIL": Image.__version__,
    "cv2": cv2.__version__,
    "pandas": pd.__version__
}

# display library versions
for lib, version in versions.items():
    print(f"{lib}: {version}")

# *****************************
# ***** Requirements File *****
# *****************************

# write dependencies to requirements.txt file
requirements = """
numpy==1.26.4
tensorflow==2.10.0
matplotlib==3.9.2
scikit-learn==1.5.2
Pillow==10.4.0
opencv-python==4.10.0
pandas==2.2.3
"""

with open("requirements.txt", "w") as file:
    file.write(requirements.strip())

# *****************************
# ***** Load Data Directories *****
# *****************************

# define base and dataset directories
base_dir = 'E:/Datas Champi'
train_dir = os.path.join(base_dir, 'balanced_train')
validation_dir = os.path.join(base_dir, 'balanced_validation')
test_dir = os.path.join(base_dir, 'test')

# check if each directory exists
for directory in [train_dir, validation_dir, test_dir]:
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

# define parameters
batch_size = 32  # batch size
img_height = 224  # image height after resizing
img_width = 224  # image width after resizing
AUTOTUNE = tf.data.AUTOTUNE  # prefetching optimization

# *****************************
# ***** Load Image Datasets *****
# *****************************

# load training dataset without rescaling
train_ds = image_dataset_from_directory(
    train_dir,
    labels='inferred',  # infer labels from sub-directory names
    label_mode='int',  # use with 'sparse_categorical_crossentropy'
    batch_size=batch_size,
    image_size=(img_height, img_width),  # resize images
    shuffle=True,  # shuffle training data
    seed=123,  # ensure reproducibility
    interpolation='bilinear'  # resample image when resizing
)

validation_ds = image_dataset_from_directory(
    validation_dir,
    labels='inferred',
    label_mode='int',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=False,  # do not shuffle validation data
    seed=123,
    interpolation='bilinear'
)

test_ds = image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='int',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=False,  # do not shuffle test data
    seed=123,
    interpolation='bilinear'
)

# *****************************
# ***** Extract Class Names *****
# *****************************

# store class names before applying .prefetch()
class_names = train_ds.class_names  # list of class names
num_classes = len(class_names)  # total number of classes
print(f"Detected classes ({num_classes}): {class_names}")  # display detected classes

# *****************************
# ***** Compute Class Weights *****
# *****************************

# initialize class count array
class_counts = np.zeros(num_classes, dtype=int)

# count samples in each class
for images, labels in train_ds:
    labels = labels.numpy()  # convert labels to numpy array
    for label in labels:
        class_counts[label] += 1  # increment class count

print(f"Samples per class: {class_counts}")

# calculate class weights for handling imbalance
y_train = np.concatenate([y.numpy() for x, y in train_ds], axis=0)  # gather all labels
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
print(f"Class weights: {class_weights_dict}")

# *****************************
# ***** Visualize Class Distribution *****
# *****************************

# create figure
plt.figure(figsize=(12, 10))

# plot bar chart for image count per class
plt.bar(class_names, class_counts, color='skyblue')

# add plot title and labels
plt.title('Image Count per Class', fontsize=16)
plt.xlabel('Classes', fontsize=14)
plt.ylabel('Image Count', fontsize=14)
plt.xticks(rotation=45, ha='right')

# annotate with image counts
for index, count in enumerate(class_counts):
    plt.text(index, count + max(class_counts)*0.01, str(count), ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.show()

# *****************************
# ***** Optimize Dataset Performance *****
# *****************************

# apply .prefetch() for dataset loading optimization
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
validation_ds = validation_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# *****************************
# ***** Utility Functions *****
# *****************************

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    return heatmap

def superimpose_heatmap(img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img_width, img_height))
    heatmap = np.uint8(255 * heatmap)
    jet = plt.get_cmap("jet")
    jet_colors = jet(heatmap)[:, :, :3]
    jet_heatmap = jet_colors * 255
    jet_heatmap = jet_heatmap.astype(np.uint8)
    superimposed_img = jet_heatmap * alpha + np.array(img)
    superimposed_img = superimposed_img.astype(np.uint8)
    superimposed_img = Image.fromarray(superimposed_img)
    return superimposed_img

# *****************************
# ***** Build Optimized CNN Model *****
# *****************************

input_shape = (img_height, img_width, 3)

model = models.Sequential([
    layers.Rescaling(1./255, input_shape=input_shape),
    layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),
    layers.Conv2D(512, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.4),
    layers.Conv2D(1024, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    layers.BatchNormalization(),
    layers.GlobalMaxPooling2D(),
    layers.Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.summary()

# *****************************
# ***** Compile Model *****
# *****************************

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# *****************************
# ***** Callbacks Definition *****
# *****************************

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        patience=10,
        restore_best_weights=True,
        monitor='val_loss'
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        save_best_only=True,
        monitor='val_loss'
    )
]

# *****************************
# ***** Model Training *****
# *****************************

history = model.fit(
    train_ds,
    epochs=20,
    validation_data=validation_ds,
    callbacks=callbacks,
    class_weight=class_weights_dict
)
