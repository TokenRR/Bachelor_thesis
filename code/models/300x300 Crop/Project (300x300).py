# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 23:11:17 2023

@author: user
"""

#%%

# Preprocessing (Y-channel):

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" 
from keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from IPython.display import display
import tensorflow as tf

crop_size = 300

upscale_factor = 3

input_size = crop_size // upscale_factor

path = "C:/Users/user/.keras/datasets/BSR/BSDS500/data/images"

train_set = image_dataset_from_directory(
    path,
    image_size=(crop_size, crop_size),
    validation_split=0.2,
    subset='training',
    seed=1337,
    label_mode=None)

valid_set = image_dataset_from_directory(
    path,
    image_size=(crop_size, crop_size),
    validation_split=0.2,
    subset='validation',
    seed=1337,
    label_mode=None)

def rescaling(input_image):
    input_image = input_image / 255.0
    return input_image
    

train_set = train_set.map(rescaling)
valid_set = valid_set.map(rescaling)

"""
for batch in train_set.take(1):
    for img in batch:
        display(array_to_img(img))     
"""
        
test_path = os.path.join(path,'test')       

test_img_paths = sorted([os.path.join(test_path,img_name)
                        for img_name in os.listdir(test_path)
                        if img_name.endswith('.jpg')])
    
shape_prior = list(train_set.as_numpy_iterator())

def process_input(input, input_size):
    input = tf.image.rgb_to_yuv(input)
    last_axis = len(input.shape) -1
    y, u, v = tf.split(input, 3, axis=last_axis)
    return tf.image.resize(y, [input_size, input_size], method="area")

def process_target(input):
    input = tf.image.rgb_to_yuv(input)
    last_axis = len(input.shape) -1
    y, u, v = tf.split(input, 3, axis=last_axis)
    return y

train_set = train_set.map(lambda x: (process_input(x, input_size),process_target(x)))

train_set = train_set.prefetch(buffer_size=32)

shape_after = list(train_set.as_numpy_iterator())

valid_set = valid_set.map(lambda x: (process_input(x, input_size),process_target(x)))

valid_set = valid_set.prefetch(buffer_size=32)

channels = shape_after[0][0].shape[-1]

"""
for batch in train_set.take(1):
    for img in batch[0]:
        display(array_to_img(img))
    for img in batch[1]:
        display(array_to_img(img))
"""

#%%

# Model (Y-channel):

from keras.layers import Conv2D
import keras
from keras.callbacks import EarlyStopping

def Model(channels, upscale_factor):

    inputs = keras.Input(shape=(None, None, channels))
    
    X = Conv2D(64, 5, padding='same', activation='relu', kernel_initializer='Orthogonal')(inputs)
    
    X = Conv2D(64, 3, padding='same', activation='relu', kernel_initializer='Orthogonal')(X)
    
    X = Conv2D(32, 3, padding='same', activation='relu', kernel_initializer='Orthogonal')(X)
    
    X = Conv2D(channels * (upscale_factor**2), 3, padding='same', activation='relu', kernel_initializer='Orthogonal')(X)
    
    outputs = tf.nn.depth_to_space(X, upscale_factor)
    
    return keras.Model(inputs, outputs)

early_stopping = EarlyStopping(monitor='loss',patience=10, min_delta=0.0001)

model = Model(channels, upscale_factor)

model.summary()

model.compile(optimizer='adam', loss='MSE')

model.fit(train_set, epochs=40, callbacks=[early_stopping], validation_data = valid_set)

#%%

# Validation (Y-channel):

from keras.models import load_model
import numpy as np
import PIL
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import matplotlib.pyplot as plt

model = load_model('Y_Model.h5')

def plot_results(img, prefix, title):
    """Plot the result with zoom-in area."""
    img_array = img_to_array(img)
    img_array = img_array.astype("float32") / 255.0

    # Create a new figure with a default 111 subplot.
    figure, parent = plt.subplots()
    parent.imshow(img_array[::-1], origin="lower")     # The reason why we want to
                                                       # plot the figure in reverse
    plt.title(title)                                   # here is because when we use
    # zoom-factor: 2.0, location: upper-left           # img_to_array, the height and
    inset = zoomed_inset_axes(parent, 2, loc='upper left') # width locations get inverted.
    inset.imshow(img_array[::-1], origin="lower")

    # Specify the limits.
    x1, x2, y1, y2 = 200, 300, 100, 200
    # Apply the x-limits.
    inset.set_xlim(x1, x2)
    # Apply the y-limits.
    inset.set_ylim(y1, y2)

    plt.yticks(visible=False)
    plt.xticks(visible=False)

    # Make the line.
    mark_inset(parent, inset, loc1=1, loc2=3, fc="none", ec="blue")
    plt.show()

for index,_ in enumerate(test_img_paths[120:140]):
    
    image = PIL.Image.open(_)
    
    plot_results(image, index, "high-res")
    
    lowres = image.resize((image.size[0] // upscale_factor,
                           image.size[1] // upscale_factor),
                          PIL.Image.BICUBIC)
    
    ycbcr = lowres.convert("YCbCr")         # Here, we converted the image to 
    y, cb, cr = ycbcr.split()               # YCbCr because we need the y-channel
    y = img_to_array(y)                     # for our model. The we did not convert
    y = y.astype("float32") / 255.0         # it to YUV is because we can't in PIL
                                            # and Y in YCbCr is the same as in YUV.
    input = y.reshape(1, y.shape[0], y.shape[1], y.shape[2])
    
    """
    In the previous line, we have to reshape our y channel because the input
    of the model must be: (Batch, Height, Width, Channels).
    """
    
    output = model.predict(input)
    
    output = output[0]
    
    output *= 255.0

    # Restore the image in RGB color space.
    
    output = output.clip(0, 255)
    
    output = output.reshape((output.shape[0], output.shape[1]))
    
    output = PIL.Image.fromarray(np.uint8(output))
    
    cb = cb.resize(output.size, PIL.Image.BICUBIC)
    
    cr = cr.resize(output.size, PIL.Image.BICUBIC)
    
    output = PIL.Image.merge("YCbCr", (output, cb, cr))
    
    output = output.convert("RGB")
    
    lowres = lowres.resize(output.size, PIL.Image.BICUBIC)
    
    plot_results(output, index, "upscaled")
    
    plot_results(lowres, index, "lowres")
    
