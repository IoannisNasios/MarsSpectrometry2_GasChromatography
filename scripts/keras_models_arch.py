# -*- coding: utf-8 -*-


import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization #Flatten, 
from tensorflow.keras.layers import Reshape, Concatenate#, LSTM, Bidirectional, 
from tensorflow.keras.layers import MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.models import Model
import gc
    
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D

from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2#, EfficientNetB3
from tensorflow.keras.layers import RandomTranslation, RandomContrast


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)



def get_model_simple(num_outputs=9, dim=(2501,)):
    """
    simple keras model architecture for training on statistical features
    Args:
        num_outputs: number of outputs - classes
        dim: number of training features
    Returns:
        a keras model
    """
    
    tf.keras.backend.clear_session()
    gc.collect()

    input_layer = Input(shape=(dim), name='input_layer')
    
    x = Dense(400, activation='swish')(input_layer)
    x = BatchNormalization()(x)

    x = Dense(300, activation='swish')(x)
    x = BatchNormalization()(x)

    x = Dense(200, activation='swish')(x)
    x = BatchNormalization()(x)

    output = Dense(num_outputs, activation='sigmoid')(x)
    model = Model(input_layer, output)
    return model




def get_model_cnn(num_outputs=9, dim=(500,500), EFF=0):
    """
    A CNN keras model with a pretrained in imagenet backbone.
    Args:
        num_outputs: number of outputs - classes
        dim: sample shape (num timesteps, num ions)
        EFF: one of [0,1,2] for which efficientnetB[] backbone to be used.
    Returns:
        a keras CNN model
    """
    img_input = Input(shape=dim)

    #augmentation layers
    x = RandomTranslation(height_factor=0.0, width_factor=0.05,  fill_mode="constant",
         interpolation="bilinear", seed=None, fill_value=0)(img_input)
    x= RandomContrast(0.1)(x)
#     x= RandomFlip(mode='horizontal')(x)
#     x= RandomBrightness(factor=0.1, value_range=(-0., 15))(x)
#    x= img_input

    #some statistical feautes
    xa = GlobalMaxPooling1D(data_format='channels_last')(img_input)
    xb = GlobalAveragePooling1D(data_format='channels_last')(img_input)
    xc = GlobalAveragePooling1D(data_format='channels_first')(img_input)
    xabc = Concatenate()([xa, xb, xc])
    xabc = Dense(512, activation='relu')(xabc)

    # Take maximum value in pool_size bin
#     x = AveragePooling1D(pool_size=3, data_format="channels_last")(img_input) # smooth
    x = MaxPooling1D(pool_size=10, data_format="channels_last")(x)
    x = Reshape((50,500,1))(x)
    
    # Repeat sample 3 times to make a 3 channels image
    img_conc = Concatenate()([x, x, x])    
    
    # Use pretrained model
    if EFF==0:
        base_model = EfficientNetB0(weights='imagenet', include_top=False,
                                    input_tensor=img_conc        )
    elif EFF==1:
        base_model = EfficientNetB1(weights='imagenet', include_top=False,
                                    input_tensor=img_conc        )
    elif EFF==2:
        base_model = EfficientNetB2(weights='imagenet', include_top=False,
                                    input_tensor=img_conc        )
#    elif EFF==3:
#        base_model = EfficientNetB3(weights=pretrain, include_top=False,
#                                    input_tensor=img_conc        )
        
    x = base_model.output

  
    x0 = GlobalMaxPooling2D()(x)
    x1 = GlobalAveragePooling2D()(x)
    x = Concatenate()([x0, x1, xabc])
    x = Dropout(0.25)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(num_outputs, activation='sigmoid')(x)

    model = Model(inputs=img_input, outputs=predictions)
    return model