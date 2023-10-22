"""
@author: Jiaxin Ye
@contact: jiaxin-ye@foxmail.com
"""
import keras.backend as K  # K.reverse(inputs, axes=1) K.concatenate([output_2, item], axis=-2)
import tensorflow as tf
from keras.layers import Activation
from keras.layers import Conv1D, SpatialDropout1D, add, GlobalAveragePooling1D
from keras.activations import sigmoid

from keras import layers
from keras import regularizers
from keras.layers import Layer, BatchNormalization, Conv2D, Lambda, Add, Concatenate, GlobalAveragePooling2D, Dense, Multiply
import math
from keras.layers import Conv2D, Lambda, BatchNormalization, AveragePooling2D, ReLU, concatenate, Input, Dropout, GlobalMaxPooling1D, MaxPooling2D, DepthwiseConv1D
from keras import backend as K

from keras.applications import ResNet50
from keras.layers import Input, Dense, Flatten
from keras.models import Model


def se_weight_block(input_tensor, c=1):
    num_channels = int(input_tensor.shape[1])  # 16
    # bottleneck = int(num_channels // c)  # 4

    input_tensor = tf.transpose(input_tensor, [0, 2, 1])

    se_branch = GlobalAveragePooling1D()(input_tensor)  # (None, 40, 16) -> (None, 16)

    se_branch = Dense(num_channels, use_bias=False, activation='sigmoid')(se_branch)  # (None, 16)

    # TODO
    # out = Multiply()([input_tensor, se_branch])  # (None, 188, 40) * (None, 40) -> (None, 188, 40)
    out = Multiply()([input_tensor, se_branch])
    out = tf.transpose(out, [0, 2, 1])
    out = GlobalAveragePooling1D()(out)


    return out


def eca_block(inputs, b=1, gamma=2):
    # Get the number of channels
    in_channels = inputs.shape[-1]  # (None, 188, 40)

    # Calculate the adaptive kernel size based on the channel dimension
    kernel_size = int(abs((math.log(in_channels, 2) + b) / gamma))

    # Ensure the kernel size is even
    if kernel_size % 2:
        kernel_size = kernel_size
    else:
        kernel_size = kernel_size + 1

    # Perform Global Average Pooling along the channel dimension
    x = tf.reduce_mean(inputs, axis=1, keepdims=True)  # (None, 1, 40) (None, 40, 16) -> (None,1, 16)

    # 1D Convolution
    x = tf.keras.layers.Conv1D(filters=1, kernel_size=kernel_size, padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(1e-6))(x)  # (None, 1, 16) -> (None, 1, 16)

    # Sigmoid activation
    x = tf.nn.sigmoid(x)

    # Feature fusion with element-wise multiplication along the channel dimension
    outputs = tf.multiply(inputs, x)  # (None, 40, 16) * (None, 1, 16) -> (None, 40, 16)

    return outputs


class WeightLayer(Layer):
    def __init__(self, **kwargs):
        super(WeightLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], 1),
                                      initializer='uniform',
                                      trainable=True)
        super(WeightLayer, self).build(input_shape)

    def call(self, x):
        tempx = tf.transpose(x, [0, 2, 1])  # (None, 16, 40) -> (None, 40, 16)
        x = K.dot(tempx, self.kernel)  # (None, 40, 16) * (None, 16, 1) -> (None, 40, 1)
        x = tf.squeeze(x, axis=-1)  # (None, 40)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])


def Temporal_Aware_Block(x, s, i, activation, nb_filters, kernel_size, dropout_rate=0, name=''):

    original_x = x  # (None, 188, 39)
    # 1.1
    conv_1_1 = Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=i, padding='causal')(x)
    conv_1_1 = BatchNormalization(trainable=True, axis=-1)(conv_1_1)
    conv_1_1 = Activation(activation)(conv_1_1)
    output_1_1 = SpatialDropout1D(dropout_rate)(conv_1_1)
    # 2.1
    conv_2_1 = Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=i, padding='causal')(output_1_1)
    conv_2_1 = BatchNormalization(trainable=True, axis=-1)(conv_2_1)
    conv_2_1 = Activation(activation)(conv_2_1)
    output_2_1 = SpatialDropout1D(dropout_rate)(conv_2_1)
    
    if original_x.shape[-1] != output_2_1.shape[-1]:
        original_x = Conv1D(filters=nb_filters, kernel_size=1, padding='same')(original_x)
        
    output_2_1 = Lambda(sigmoid)(output_2_1)
    F_x = Lambda(lambda x: tf.multiply(x[0], x[1]))([original_x, output_2_1])  # original_x 和 output_2_1都是 (None, 188, 39)
    return F_x


class TIMNET:
    def __init__(self,
                 nb_filters=64,
                 kernel_size=2,
                 nb_stacks=1,
                 dilations=None,
                 activation="relu",
                 dropout_rate=0.1,
                 return_sequences=True,
                 name='TIMNET'):
        self.name = name
        self.return_sequences = return_sequences
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters

        self.supports_masking = True
        self.mask_value = 0.

        if not isinstance(nb_filters, int):
            raise Exception()

    def __call__(self, inputs, mask=None):
        if self.dilations is None:
            self.dilations = 8
        forward = inputs
        backward = K.reverse(inputs, axes=1)

        print("Input Shape=", inputs.shape)
        forward_convd = Conv1D(filters=self.nb_filters, kernel_size=1, dilation_rate=1, padding='causal')(forward)
        backward_convd = Conv1D(filters=self.nb_filters, kernel_size=1, dilation_rate=1, padding='causal')(backward)

        final_skip_connection = []

        skip_out_forward = forward_convd  # （None, 188, 39)
        skip_out_backward = backward_convd  # （None, 188, 39)

        for s in range(self.nb_stacks):
            for i in [2 ** i for i in range(self.dilations)]:
                skip_out_forward = Temporal_Aware_Block(skip_out_forward, s, i, self.activation,
                                                        self.nb_filters,
                                                        self.kernel_size,
                                                        self.dropout_rate,
                                                        name=self.name)  # （None, 188, 39)
                skip_out_backward = Temporal_Aware_Block(skip_out_backward, s, i, self.activation,
                                                         self.nb_filters,
                                                         self.kernel_size,
                                                         self.dropout_rate,
                                                         name=self.name)  # （None, 188, 39)

                temp_skip = add([skip_out_forward, skip_out_backward],
                                name="biadd_" + str(i))  # temp_skip:（None, 188, 39)

                # TODO
                temp_skip = eca_block(temp_skip)

                temp_skip = GlobalAveragePooling1D()(temp_skip)  # （None, 39)
                temp_skip = tf.expand_dims(temp_skip, axis=1)  # （None, 1, 39)
                final_skip_connection.append(temp_skip)  # list:8

        output_2 = final_skip_connection[0]  # （None, 1, 39)
        for i, item in enumerate(final_skip_connection):
            if i == 0:
                continue
            output_2 = K.concatenate([output_2, item], axis=-2)
        x2 = output_2  # (None, 8, 40)

        # TODO 试试SE代替Weight
        x = se_weight_block(x2)  # (None, 40)

        x = Dense(2, activation='softmax')(x)  # (None, 2)

        return x
