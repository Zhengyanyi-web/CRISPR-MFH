import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras.models import Model
from keras_multi_head import MultiHeadAttention
from tensorflow.keras.layers import Add, LeakyReLU, GRU,Embedding,Activation,ReLU,Multiply,AveragePooling2D,MaxPool2D,MaxPooling1D,BatchNormalization,Conv1D,\
    Attention, Dense, Conv2D, Bidirectional, LSTM,Dot, DepthwiseConv2D, MaxPooling2D,GlobalAveragePooling2D, Flatten, Input, Activation, Reshape, Dropout, Concatenate, AveragePooling1D, MaxPool1D, \
    BatchNormalization, Attention,RepeatVector, GlobalAveragePooling1D, GlobalMaxPool1D, GRU, AdditiveAttention, AlphaDropout, LeakyReLU
from tensorflow.keras.initializers import VarianceScaling,RandomUniform
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, average_precision_score
from keras_layer_normalization import LayerNormalization
from tensorflow.keras.initializers import glorot_normal, HeNormal
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import multiply, concatenate
from tensorflow.python.keras.layers.core import Dense, Dropout, Lambda, Flatten, Reshape, Permute
from tensorflow.keras import layers, models
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def DownSample_new(inputs, filters, kernel_size=3, strides=2, padding='same'):
    """
    Downsamples the input by using a convolutional layer with strides.

    Parameters:
    inputs (tensor): Input tensor.
    filters (int): Number of filters for the convolution.
    kernel_size (int or tuple): Size of the convolutional kernel.
    strides (int or tuple): Strides for the convolution.
    padding (str): Padding type, either 'valid' or 'same'.

    Returns:
    tensor: Output tensor after downsampling.
    """
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding)(inputs)
    x = BatchNormalization()(x)
    x = Activation(swish)(x)
    return x


def gDP_block(inputs, filters, kernel_sizes=[1, 3, 5, 7]):
    inputs_conv = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    depthwise_convs = [DepthwiseConv2D(kernel_size=ks, padding='same')(inputs_conv) for ks in kernel_sizes]
    x = tf.concat(depthwise_convs, axis=-1)
    x = Conv2D(filters, kernel_size=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Add skip connection
    x = Add()([inputs, x])
    return x


def channel_attention(input_tensor, ratio=8):
    channel_axis = -1
    channel = input_tensor.shape[channel_axis]

    shared_layer_one = Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_tensor)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = GlobalAveragePooling2D()(input_tensor)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    cbam_feature = tf.keras.layers.Add()([avg_pool, max_pool])
    cbam_feature = tf.keras.layers.Activation('sigmoid')(cbam_feature)

    return Multiply()([input_tensor, cbam_feature])


def spatial_attention(input_tensor):
    kernel_size = 7
    avg_pool = tf.reduce_mean(input_tensor, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(input_tensor, axis=-1, keepdims=True)
    concat = Concatenate(axis=-1)([avg_pool, max_pool])
    cbam_feature = Conv2D(filters=1, kernel_size=kernel_size, strides=1, padding='same', activation='sigmoid',
                          kernel_initializer='he_normal', use_bias=False)(concat)

    return Multiply()([input_tensor, cbam_feature])


def EPA(inputs):
    # Channel Attention
    ca = channel_attention(inputs)
    # Spatial Attention
    sa = spatial_attention(inputs)
    # Concatenate
    concat = Concatenate()([ca, sa])
    # 1x1 Convolution
    output = Conv2D(inputs.shape[-1], (1, 1), padding='same')(concat)

    return output


def CRISPR_MFH():

    input_predict = Input(shape=(1, 24, 7), name='input_predict')
    input_on = Input(shape=(1, 24, 5), name='input_on')
    input_off = Input(shape=(1, 24, 5), name='input_off')

    predictstem = DownSample(input_predict, 16, strides=1)
    onstem = DownSample(input_on, 16, strides=1)
    offstem = DownSample(input_off, 16, strides=1)

    concat_output = concatenate([predictstem, onstem, offstem], axis=-1)

    msplck_output = gDP_block(concat_output,48)
    final_output = Add()([concat_output, msplck_output])

    epa_output = EPA(final_output)

    final_epa_output = Add()([final_output, epa_output])

    blstm_out = Flatten()(final_epa_output)

    x = Dense(80, kernel_initializer=glorot_normal(seed=2024), activation='relu')(blstm_out)
    x = Dense(20, kernel_initializer=glorot_normal(seed=2024), activation='relu')(x)
    x = Dropout(0.35, seed=2024)(x)
    output = Dense(2, kernel_initializer=glorot_normal(seed=2024), activation='softmax', name='main_output')(x)

    model = Model(inputs=[input_predict, input_on, input_off], outputs=[output])
    return model

if __name__ == '__main__':
    # pass
    mdoel = lcx()
    mdoel.summary()
    pass


