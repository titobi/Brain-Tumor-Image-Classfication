# Brain-Tumor-Image-Classfication
A Brain Tumour is a collection of abnormal tissues found in the brain, spinal cord, covering of the brain or nerves leading from the brain. Without early detection it can be life-threatening to the patient. In this project, i tested Deep learning models and checked if it would improve the accuracy and efficiency of brain tumour classification in MRI images

## Tools
- Goggle Colab Notebook,
- Python 3.7+

## File Description
The dataset was downloaded from [Here](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset?select)

This dataset contains 7023 images of human brain MRI images which are classified into 4 classes
- Glioma
- Meningioma
- No tumor
- Pituitary

## Import Packages
```ruby
import os,glob,sys
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from sklearn import model_selection, metrics, preprocessing
from tqdm import tqdm
import pandas as pd
from keras.utils import to_categorical
from keras.applications.resnet import ResNet50
```

## Analysis
Distribution of the categories of brain tumour MRI images in the traning data
![tumour chat](https://github.com/user-attachments/assets/7ecab5dd-1cc6-4548-aa64-2deff8caff7b)

### Data Preprocessing
```ruby
# Convert lists to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Normalize image data
X_train = X_train / 255.0
X_test = X_test / 255.0
```
```ruby
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Initialize the label encoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_train = to_categorical(y_train, num_classes=len(np.unique(y_train)))

y_test = le.transform(y_test)
y_test = to_categorical(y_test, num_classes=len(np.unique(y_test)))

# Check the new shapes
print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)
```
### Data Augumentation
```ruby

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the ImageDataGenerator for the images with augmentations
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)
```
## Evaluation of models
### VGG16
```ruby
# Load the VGG16 model
base_model = tf.keras.applications.vgg16.VGG16(include_top=False,
                                               weights='imagenet',
                                               input_shape=(224, 224, 3),
                                               pooling='avg')

```
### ResNet50
```ruby
#Load the ResNet50 model
base_model = tf.keras.applications.resnet.ResNet50(include_top=False,
                                               weights='imagenet',
                                               input_shape=(224, 224, 3),
                                               pooling='avg')
```
### Custom CNN
```ruby
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

```
## Result
### Confusion Matrix of VGG16
![download (11)](https://github.com/user-attachments/assets/b9f7da3f-a1db-4ec1-a2ac-929d8643c541)

### Confusion Matrix of ResNet50
![download (12)](https://github.com/user-attachments/assets/57cda0c8-3fed-4253-8bcf-3a969196f129)

### Confusion Matrix of Custom CNN
![download (13)](https://github.com/user-attachments/assets/9044b544-380f-48df-95f8-63325d9fcc76)

