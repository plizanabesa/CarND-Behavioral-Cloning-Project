import pickle
import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import cv2
import scipy
import sys
import os
import pandas
import argparse
import json

from PIL import Image

from keras.layers import Input, Flatten, Dense, Lambda, ELU, Dropout, SpatialDropout2D
from keras.models import Model, Sequential
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras import initializations

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from skimage import transform as trf

# Augment data region
def augment_image_brightness(image):
    new_image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    new_image[:,:,2] = new_image[:,:,2]*random_bright
    new_image = cv2.cvtColor(new_image,cv2.COLOR_HSV2RGB)
    return new_image

def shift_image(image,steer,shift_range):
    shift_x = shift_range*np.random.uniform()-shift_range/2
    new_steer = steer + shift_x/shift_range*2*.2
    shift_y = 40*np.random.uniform()-40/2
    Trans_M = np.float32([[1,0,shift_x],[0,1,shift_y]])
    new_image = cv2.warpAffine(image,Trans_M,(320,160)) 
    return new_image,new_steer

def augment_image_shadow(image, ):
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright    
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image

def flip_image(image,steer):
    random_flip = np.random.randint(2)
    new_image = image
    new_steer = steer
    if random_flip == 0:
      new_image = cv2.flip(image,1)
      new_steer = -steer
    return new_image,new_steer

def gen_old(data, data_size, batch_size): 
  first=0
  last=first+batch_size
  while True:
    X_batch = np.array([mpimg.imread(os.path.join('data',image)) for image in data[first:last,0]])
    #print(X_batch.shape)
    y_batch = np.array(data[first:last,1])
    first += batch_size
    last += batch_size
    if last >= data_size:
      first = 0  
      last = first + batch_size
    #print(first,last)    
    yield (X_batch, y_batch)

def gen(data, data_size, batch_size): 
  X_batch = np.zeros((batch_size, 160, 320, 3))
  y_batch = np.zeros(batch_size)
  while True:
    for i in range(batch_size):
      rand_line = np.random.randint(data_size)
      keep_iter = 0
      while keep_iter==0:
        image_line = data[rand_line,0]
        image = mpimg.imread(os.path.join('data',image_line))
        steer = data[rand_line,1]
        #shift image
        image,steer = shift_image(image,steer,100)
        #augment image brightness
        image = augment_image_brightness(image)
        #augment image shadow
        image = augment_image_shadow(image)
        #flip image
        image,steer = flip_image(image,steer)

        if abs(steer)<0.1:
          prob = np.random.uniform()
          if prob > 0.5:
            keep_iter=1
        else:
          keep_iter=1

      X_batch[i]=image
      y_batch[i]=steer
      #plt.imshow(image)
      #print(steer)
      yield (X_batch, y_batch)

# Preprocessing data region
def resize(image):
    import tensorflow as tf #Import in here to be used by the model drive.py
    return tf.image.resize_images(image,(80, 160))

def resize_invidia(image):
    import tensorflow as tf #Import in here to be used by the model drive.py
    return tf.image.resize_images(image,(66, 200))

def resize_blog(image):
    import tensorflow as tf #Import in here to be used by the model drive.py
    return tf.image.resize_images(image,(64, 64))

def normalize_greyscale(image):
    a = -0.5
    b = 0.5
    grayscale_min = 0
    grayscale_max = 255
    return a + ( ( (image - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )

def crop_image(image):
    return image[:, :, 22:135, :]

#Model definition region
def get_comma_ai_model():
  ch, row, col = 3, 160, 320  # camera format
  model = Sequential()
  model.add(Cropping2D(cropping=((25,0),(25,0)),input_shape=(row, col, ch)))
  model.add(Lambda(resize))
  model.add(Lambda(normalize_greyscale)) # different normalization ([-0.5,0.5] range) than original comma_ai model ([-1,1] range)
  #model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=(row, col, ch)))
  model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
  model.add(ELU())
  model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
  model.add(ELU())
  model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
  model.add(Flatten())
  model.add(Dropout(.2))
  model.add(ELU())
  model.add(Dense(512))
  model.add(Dropout(.5))
  model.add(ELU())
  model.add(Dense(1))

  model.compile(optimizer="adam", loss="mse", metrics=['mse'])
  print (model.summary())

  return model

def get_invidia_model(learning_rate):
  ch, row, col = 3, 160, 320  # camera format

  model = Sequential()

  model.add(Cropping2D(cropping=((30,0),(25,0)),input_shape=(row, col, ch)))
  #model.add(Cropping2D(crop_image,input_shape=(row, col, ch)))
  model.add(Lambda(resize_invidia))
  #model.add(Lambda(resize_invidia,input_shape=(row, col, ch)))
  model.add(Lambda(normalize_greyscale))
  
  model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="valid", init= initializations.uniform))
  model.add(ELU())
  model.add(SpatialDropout2D(.5))
  
  model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="valid", init= initializations.uniform))
  model.add(ELU())
  model.add(SpatialDropout2D(.5))

  model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="valid", init= initializations.uniform))
  model.add(ELU())
  model.add(SpatialDropout2D(.5))

  model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init= initializations.uniform))
  model.add(ELU())
  model.add(SpatialDropout2D(.5))

  model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init= initializations.uniform))
  model.add(ELU())
  model.add(SpatialDropout2D(.5))
  
  model.add(Flatten())
  
  model.add(Dense(100, init= initializations.uniform))
  model.add(ELU())
  model.add(Dropout(.5))
  
  model.add(Dense(50, init= initializations.uniform))
  model.add(ELU())
  #model.add(Dropout(.5))
  
  model.add(Dense(10, init= initializations.uniform))
  model.add(ELU())
  model.add(Dropout(.5))

  model.add(Dense(1))

  model.compile(optimizer=Adam(lr=learning_rate), loss="mse", metrics=['mse'])
  print (model.summary())

  return model

def get_blog_model(learning_rate):
  ch, row, col = 3, 160, 320  # camera format

  model = Sequential()

  model.add(Cropping2D(cropping=((30,0),(25,0)),input_shape=(row, col, ch)))
  #model.add(Cropping2D(crop_image,input_shape=(row, col, ch)))
  model.add(Lambda(resize_blog))
  #model.add(Lambda(resize_invidia,input_shape=(row, col, ch)))
  model.add(Lambda(normalize_greyscale))
  
  model.add(Convolution2D(3, 1, 1))

  model.add(Convolution2D(32, 3, 3))
  model.add(Convolution2D(32, 3, 3))
  model.add(MaxPooling2D((2, 2)))
  model.add(ELU())
  model.add(SpatialDropout2D(.5))

  model.add(Convolution2D(64, 3, 3))
  model.add(Convolution2D(64, 3, 3))
  model.add(MaxPooling2D((2, 2)))
  model.add(ELU())
  model.add(SpatialDropout2D(.5))

  model.add(Convolution2D(128, 3, 3))
  model.add(Convolution2D(128, 3, 3))
  model.add(MaxPooling2D((2, 2)))
  model.add(ELU())
  model.add(SpatialDropout2D(.5))

  model.add(Flatten())
  
  model.add(Dense(512))
  model.add(ELU())

  model.add(Dense(64))
  model.add(ELU())

  model.add(Dense(16))
  model.add(Dense(1))

  model.compile(optimizer=Adam(lr=learning_rate), loss="mse", metrics=['mse'])
  print (model.summary())

  return model

def get_model(learning_rate):
  ch, row, col = 3, 160, 320  # camera format

  model = Sequential()

  model.add(Cropping2D(cropping=((30,0),(25,0)),input_shape=(row, col, ch)))
  #model.add(Cropping2D(crop_image,input_shape=(row, col, ch)))
  model.add(Lambda(resize_blog))
  #model.add(Lambda(resize_invidia,input_shape=(row, col, ch)))
  model.add(Lambda(normalize_greyscale))
  
  model.add(Convolution2D(3, 1, 1))

  model.add(Convolution2D(16, 3, 3))
  model.add(Convolution2D(16, 3, 3))
  model.add(MaxPooling2D((2, 2)))
  model.add(ELU())
  model.add(SpatialDropout2D(.5))

  model.add(Convolution2D(32, 3, 3))
  model.add(Convolution2D(32, 3, 3))
  model.add(MaxPooling2D((2, 2)))
  model.add(ELU())
  model.add(SpatialDropout2D(.5))

  model.add(Convolution2D(64, 3, 3))
  model.add(Convolution2D(64, 3, 3))
  model.add(MaxPooling2D((2, 2)))
  model.add(ELU())
  model.add(SpatialDropout2D(.5))

  model.add(Flatten())
  
  model.add(Dense(1024))
  model.add(ELU())

  model.add(Dense(64))
  model.add(ELU())

  model.add(Dense(16))
  model.add(Dense(1))

  model.compile(optimizer=Adam(lr=learning_rate), loss="mse", metrics=['mse'])
  print (model.summary())

  return model

if __name__ == "__main__":
  #parser.add_argument('--batch', type=int, default=75, help='Batch size.')
  #parser.add_argument('--epoch', type=int, default=200, help='Number of epochs.')
  batch = 200
  epoch = 8
  learning_rate = 0.0001
  p_data = pd.read_csv(os.path.join('data','driving_log.csv'), sep=None, engine='python')
  train, val = train_test_split(p_data, test_size=0.3, random_state=10)
  val, test = train_test_split(val, test_size=0.3, random_state=10)

  #train data
  center_data = np.column_stack((np.array((train.center)),np.array((train.steering))))
  left_data = np.column_stack((np.array((train.left)),np.array((train.steering+0.2)))) # add 0.2 to the steering angle
  right_data = np.column_stack((np.array((train.right)),np.array((train.steering-0.2)))) # substract 0.2 to the steering angle
  train_data = np.vstack((center_data,left_data,right_data))
  #train_data = center_data

  #validation data
  val_data = np.column_stack((np.array((val.center)),np.array((val.steering))))

  #test data
  test_data = np.column_stack((np.array((test.center)),np.array((test.steering))))

  #shuffle data: not needed anymore since in the generator I randomly pick up images from the data
  #np.random.shuffle(train_data)
  #np.random.shuffle(val_data)
  #np.random.shuffle(test_data)

  #model = get_comma_ai_model()
  #model = get_invidia_model(learning_rate)
  model = get_blog_model(learning_rate)
  #model = get_model(learning_rate)
  print('{}: {}'.format("Train samples",len(train_data)))
  print('{}: {}'.format("Validation samples",len(val_data)))
  print('{}: {}'.format("Test samples",len(test_data)))

  model.fit_generator(
      gen(train_data, len(train_data), batch),
      samples_per_epoch = 25000, #30000
      #samples_per_epoch = len(train_data),
      nb_epoch= epoch,
      validation_data = gen(val_data, len(val_data), batch),
      nb_val_samples = 8000 #7500
  )

  #serialize model to JSON
  model_json = model.to_json()
  with open("model.json", "w") as json_file:
    json.dump(model_json, json_file)
  #serialize weights to HDF5
  model.save_weights("model.h5")

  #evalute model on test set
  X_test = np.array([mpimg.imread(os.path.join('data',image)) for image in test_data[:,0]])
  y_test = np.array(test_data[:,1])
  metrics = model.evaluate(X_test, y_test, batch_size=batch)
  for metric_i in range(len(model.metrics_names)):
    metric_name = model.metrics_names[metric_i]
    metric_value = metrics[metric_i]
    print('{}: {}'.format(metric_name, metric_value))

  #predict steering angles in test set
  predictions = model.predict(X_test, batch_size = batch)
  print(predictions)
  print(y_test)

  """
  X_train = np.array([mpimg.imread(os.path.join('data_test',image)) for image in train_data[:,0]])
  y_train = np.array(train_data[:,1])
  predictions = model.predict(X_train, batch_size = batch)
  print(predictions)
  print(y_train)
  """




