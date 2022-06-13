# Copyright © 2021 Ronaldson Bellande
from __future__ import print_function
import cv2, sys, math, random, warnings, os, os.path, json, pydicom, glob, shutil, datetime, zipfile, urllib.request, tensorflow as tf, time, trimesh, librosa, gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from os.path import basename
from PIL import Image, ImageDraw
from imgaug import augmenters as iaa
from tqdm import tqdm
from random import randint
import image_slicer
from numpy import expand_dims
from contextlib import redirect_stdout
from gym import error, spaces, utils
from multiprocessing import Pool

import nvidia_smi
from os import listdir
from xml.etree import ElementTree
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as img
from collections import deque

from sklearn.tree import DecisionTreeRegressor
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import RobustScaler, StandardScaler, LabelEncoder, MinMaxScaler, OneHotEncoder, LabelBinarizer
from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error,  classification_report, confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, average_precision_score, auc, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, KFold, cross_val_predict, StratifiedKFold, train_test_split, learning_curve, ShuffleSplit, train_test_split,  KFold, cross_val_score, StratifiedShuffleSplit
from sklearn import model_selection, preprocessing
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier

from mrcnn import utils, visualize
# import mrcnn.model as modellib
from mrcnn.config import Config
# from mrcnn import model as modellib, utils
from mrcnn.visualize import display_images, display_instances
# from mrcnn.model import log
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

from tensorflow.python.client import device_lib
from tensorflow import convert_to_tensor
import tensorflow.keras
import keras.backend as K

from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import mixed_precision, Model
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout, Activation, LSTM
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

warnings.filterwarnings('ignore')
plt.style.use('ggplot')


from gpu_cpu_efficiency import *
gpu_enable()
mixed_precision()

from all_models import *
from speech_recognition_system.speech_utilities import *
from speech_recognition_system.plot_and_animation import *
from speech_recognition_system.speech_model_building import *
from speech_recognition_system.speech_model_training import *
from speech_recognition_system.speech_model_transfer_learning import *
from speech_recognition_system.speech_model_classification import *
from speech_recognition_system.deep_learning_model import *
from speech_recognition_system.speech_enviroment import *
from speech_recognition_system.speech_continuous_learning import *


from speech_pitch.pitch_shift import *
