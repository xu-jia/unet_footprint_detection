"""
U-net encoder from Pre-trained models.
create a functional model that outputting feature extraction layers
"""
import tensorflow as tf
from tensorflow.keras.applications import VGG16,ResNet50,MobileNetV2
def unet_vgg16():
    base_model = VGG16(include_top=False,weights="models/model_weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
    layer_names = [
        'block1_conv2',
        'block2_conv2',
        'block3_conv2',
        'block4_conv2',
        'block5_conv3'
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]
    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs, name = "encode_model_vgg16")
    return down_stack
def unet_resnet50():
    base_model = ResNet50(include_top=False, weights="models/model_weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")
    layer_names = [
        "conv1_relu",
        "conv2_block1_1_relu",
        "conv3_block1_1_relu",
        "conv4_block1_1_relu",
        "conv5_block1_1_relu",
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]
    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs, name = "encode_model_resnet50")
    return down_stack

def unet_mobilenet():
    base_model = MobileNetV2(include_top=False,weights="models/model_weights/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5")
    layer_names = [
        'block_1_expand_relu',
        'block_3_expand_relu',
        'block_6_expand_relu',
        'block_13_expand_relu',
        'block_16_project'
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]
    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs, name = "encode_model_mobilenet")
    return down_stack

