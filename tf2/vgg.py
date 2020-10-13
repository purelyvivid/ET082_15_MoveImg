from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.utils import get_file
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense 
import warnings
import os
import subprocess

import tensorflow as tf

# make dir
rt = subprocess.run(["ls","model"])
if rt.returncode!=0:
    subprocess.run(["mkdir","model"])

rt = subprocess.run(["ls","model/vgg"])
if rt.returncode!=0:
    subprocess.run(["mkdir","model/vgg"])
    
rt = subprocess.run(["ls","model/vggface"])
if rt.returncode!=0:
    subprocess.run(["mkdir","model/vggface"])

# define weight path
VGGFACE_VGG16_WEIGHTS_LOAD_PATH =  \
'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_vgg16.h5'
VGGFACE_DIR = os.getcwd()+'/model/vggface/'
VGGFACE_SAVE_FILENAME = 'rcmalli_vggface_tf_vgg16.h5'

VGG19_WEIGHTS_LOAD_PATH =  \
'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5'
VGG_DIR = os.getcwd()+'/model/vgg/'
VGG_SAVE_FILENAME = 'rcmalli_vggface_tf_vgg16.h5'

def warn_channel_first():
    warnings.warn('You are using the TensorFlow backend, yet you '
          'are using the Theano '
          'image data format convention '
          '(`image_data_format="channels_first"`). '
          'For best performance, set '
          '`image_data_format="channels_last"` in '
          'your Keras config '
          'at ~/.keras/keras.json.')    
    
def get_model_VGGface():

    tf.keras.backend.set_image_data_format('channels_last')
    
    weights_path = VGGFACE_DIR+VGGFACE_SAVE_FILENAME
    if not os.path.exists(weights_path):
        # load weight file
        weights_path = get_file(VGGFACE_SAVE_FILENAME,
                        VGGFACE_VGG16_WEIGHTS_LOAD_PATH,
                        cache_subdir=VGGFACE_DIR)
    
    # build modle
    base_model = VGG16(weights=None, include_top=True) 
    x = base_model.layers[-2].output 
    predictions = Dense(2622, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # check 
    if model.layers[1].data_format == "channels_first":
        warn_channel_first()

    # load weights
    model.load_weights(weights_path)
    
    tf.keras.backend.set_image_data_format('channels_first')
    
    return model

def get_model_VGG():
    
    tf.keras.backend.set_image_data_format('channels_last')
    
    weights_path = VGG_DIR+VGG_SAVE_FILENAME
    if not os.path.exists(weights_path):
        # load weight file
        weights_path = get_file(VGG_SAVE_FILENAME,
                        VGG19_WEIGHTS_LOAD_PATH,
                        cache_subdir=VGG_DIR)
    
    # build modle
    model = VGG19(weights=None, include_top=True) 
    
    # check 
    if model.layers[1].data_format == "channels_first":
        warn_channel_first()
    
    # load weights
    model.load_weights(weights_path) 
    
    tf.keras.backend.set_image_data_format('channels_first')
    
    return model