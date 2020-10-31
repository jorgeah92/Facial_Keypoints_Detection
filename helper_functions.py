import numpy as np
from numpy import expand_dims
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output
from time import sleep
import os
from keras.layers import Conv2D,Dropout,Dense,Flatten
from keras.models import Sequential
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.layers import Activation, Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Conv2D,MaxPool2D, ZeroPadding2D

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

import tensorflow
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, utils
from torch import nn, optim
import torch.nn.functional as F

import warnings
warnings.filterwarnings('ignore')

#Read data 
#sample = pd.read_csv(data_dir + 'SampleSubmission.csv')
#train_data = pd.read_csv(data_dir + 'training.csv')
#test_data = pd.read_csv(data_dir + 'test.csv')

data_dir = 'data_files/'
lookid_data = pd.read_csv(data_dir + 'IdLookupTable.csv')
train_data1 = pd.read_parquet(data_dir + 'train_data_1.gzip')
train_data2 = pd.read_parquet(data_dir + 'train_data_2.gzip')
train_data = pd.concat([train_data1, train_data2])
test_data = pd.read_parquet(data_dir + 'test_data.gzip')


IMG_SIZE = 96 # image size 96 x 96 pixels

def show_keypoints(image, keypoints):
    '''
    Show image with keypoints
    Args:
        image (array-like or PIL image): The image data. (M, N)
        keypoints (array-like): The keypoits data. (N, 2)
    '''
      
    plt.imshow(image, cmap='gray')
    if len(keypoints):
        plt.scatter(keypoints[:, 0], keypoints[:, 1], s=24, marker ='.', c='r')
        
def show_images(df, indxs, ncols=5, figsize=(15,10), with_keypoints=True):
    '''
    Show images with keypoints in grids
    Args:
        df (DataFrame): data (M x N)
        idxs (iterators): list, Range, Indexes
        ncols (integer): number of columns (images by rows)
        figsize (float, float): width, height in inches
        with_keypoints (boolean): True if show image with keypoints
    '''
    plt.figure(figsize=figsize)
    nrows = len(indxs) // ncols + 1
    for i, idx in enumerate(indxs):
        image = np.fromstring(df.loc[idx, 'Image'], sep=' ').astype(np.float32)\
                .reshape(-1, IMG_SIZE)
        if with_keypoints:
            keypoints = df.loc[idx].drop('Image').values.astype(np.float32)\
                        .reshape(-1, 2)
        else:
            keypoints = []
        plt.subplot(nrows, ncols, i + 1)
        plt.title(f'Sample #{idx}')
        plt.axis('off')
        plt.tight_layout()
        show_keypoints(image, keypoints)
    plt.show()


def get_features(df, dim=2):
    #Input train or test dataframe and number of dimensions you want features in.
    #Returns vector of features (pixel intensities for all examples)
    #TODO: divided by 255 for scaling?
    
    images_list = []

    for i in range(0, df.shape[0]):
        image = df["Image"][i].split(' ')
        image = ["0" if x == '' else x for x in image]
        images_list.append(image)
    
    images_array = np.array(images_list, dtype="float")
    if dim==2:
        images_features = images_array.reshape(-1, 96, 96, 1)
    else:
        images_features = images_array

    return images_features

def get_labels(df):
    #Input only test dataframe
    #Returns vector of labels (num_examples by 30 column vector of X,Y coords for face keypoints)
    
    #Grabbing the corresponding training labels
    labels_df = df.drop("Image", axis = 1)
    image_labels = []

    for i in range(0, df.shape[0]):
        keypoint_coords = labels_df.iloc[i, :]
        image_labels.append(keypoint_coords)
    
    return np.array(image_labels, dtype = "float")
    

class Normalize(object):
    '''Normalize input images'''
    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        return {'image': image / 255., # scale to [0, 1]
                'keypoints': keypoints}
        

