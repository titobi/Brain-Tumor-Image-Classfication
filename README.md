# Brain-Tumor-Image-Classfication
A Brain Tumour is a collection of abnormal tissues found in the brain, spinal cord, covering of the brain or nerves leading from the brain. Without early detection it can be life-threatening to the patient

## Tools
- Goggle Colab Notebook,
- Python 3.7+

## File Description
This dataset contains 7023 images of human brain MRI images which are classified into 4 classes
--glioma
--meningioma
--no tumor
--pituitary

## Import Packages
'''ruby
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
'''
