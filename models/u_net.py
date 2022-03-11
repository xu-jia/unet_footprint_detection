"""
This module performs the function to define U-net model, which consists of an encoder and a decoder.
The encoder can be
    - a pre-trained model without top layers, ["vgg16","mobilenet","resnet50"], created by loading weights trained with large dataset (A freeze-encoder option is provided.)
    - model from scratch
"""
import tensorflow as tf
from tensorflow.keras.layers import Input,Conv2D,MaxPooling2D,BatchNormalization,Activation,concatenate,Conv2DTranspose,Dropout
from .unet_encoder_pretrain import unet_vgg16, unet_resnet50, unet_mobilenet
from .utils import freeze_model

def downsampling_block(inputs=None, n_filters=64, dropout_prob=0, BN = 0, max_pooling=True):
    """
    Convolutional downsampling block
    Arguments:
        inputs -- Input tensor
        n_filters -- Number of filters for the convolutional layers
        dropout_prob -- Dropout probability
        max_pooling -- Use MaxPooling2D to reduce the spatial dimensions of the output volume
    Returns:
        next_layer, skip_connection --  Next layer and skip connection outputs
    """
    conv = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='HeNormal')(inputs)
    if BN:
        conv = BatchNormalization()(conv)

    conv = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='HeNormal')(conv)
    if BN:
        conv = BatchNormalization()(conv)

    if dropout_prob > 0:
        conv = Dropout(dropout_prob)(conv)

    if max_pooling:
        next_layer = MaxPooling2D(pool_size=(2, 2))(conv)
    else:
        next_layer = conv

    skip_connection = conv # copy and crop the feature map
    return next_layer, skip_connection


def upsampling_block(expansive_input, contractive_input, n_filters=64):
    """
    deConvolutional upsampling block

    Arguments:
        expansive_input -- Input tensor from previous layer
        contractive_input -- Input tensor from previous skip layer
        n_filters -- Number of filters for the convolutional layers
    Returns:
        conv -- Tensor output
    """
    up = Conv2DTranspose(n_filters, 3, strides=2, padding='same')(expansive_input)

    # Merge the previous output and the contractive_input
    merge = concatenate([up, contractive_input], axis=3)
    conv = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='HeNormal')(merge)
    conv = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='HeNormal')(conv)

    return conv

def build_model(base_model = None,
               input_height = 224,
               input_width = 224,
               n_channels = 3,
               n_filters =64,
               n_classes = 2,
               dropout_prob = 0,
               encoder_freeze =False,
                **kwargs):
    """
    :param base_model_name: vgg16, mobilenet or resnet. if true, encoder network from given base model trained from large dataset
    :param n_filters: number of filters
    :param n_classes: number of classes in the output layer
    :param dropout_prob: if true, dropout in the first conv layers
    :param encoder_freeze: if true, encoder layers are not trainable
    :return:
    """
    input_shape = (input_height, input_width, n_channels)
    inputs = Input(input_shape)
    # get encoder model from pre-trained models
    if base_model:
        assert base_model in ["vgg16","mobilenet","resnet50"], "Please enter a valid model name!"
        assert n_filters == 64, "The number of filters should be 64 with pre-trained encoder option"
        if base_model == "vgg16":
            encode_model = unet_vgg16()
        if base_model == "mobilenet":
            encode_model = unet_mobilenet()
        if base_model == "resnet50":
            encode_model = unet_resnet50()
    else:
        inputs0 = Input(input_shape)
        cblock1 = downsampling_block(inputs0, n_filters, dropout_prob, BN=1)
        cblock2 = downsampling_block(cblock1[0], n_filters * 2, dropout_prob, BN=1)
        cblock3 = downsampling_block(cblock2[0], n_filters * 4, dropout_prob, BN=1)
        cblock4 = downsampling_block(cblock3[0], n_filters * 8, dropout_prob, BN=1)
        cblock5 = downsampling_block(cblock4[0], n_filters * 16, max_pooling=False)
        encode_model = tf.keras.Model(inputs = inputs0, outputs = [cblock1[1],cblock2[1],cblock3[1],cblock4[1],cblock5[0]],name = "encode_model")

    if encoder_freeze:
        freeze_model(encode_model)

    skips = encode_model(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])
    for skip, n in zip(skips, [8, 4, 2, 1]):
        x = upsampling_block(x, skip, n_filters=n_filters*n)
    if base_model == "mobilenet" or base_model == "resnet50":
        # This is the last upsampling layer to the image size
        last = Conv2DTranspose(n_filters/2, 3, strides=2, padding='same')
        x = last(x)
    x = Conv2D(n_classes, 1, activation="softmax", padding='same')(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model




