from tensorflow.keras import backend as K
import tensorflow as tf

def iou_coef(y_true,y_pred,smooth=1e-6):
    y_pred = K.cast(K.argmax(y_pred, axis=-1),tf.float32)[...,tf.newaxis]
    intersection = K.sum(y_true*y_pred,axis = [1,2,3])
    union = K.sum(y_true+y_pred,axis = [1,2,3])-intersection
    iou = (intersection+smooth) / (union + smooth)
    return K.mean(iou)

def dice_coef(y_true, y_pred,  smooth=1e-6):
    y_pred = K.cast(K.argmax(y_pred, axis=-1),tf.float32)[...,tf.newaxis]
    intersection = K.sum(y_true*y_pred,axis = [1,2,3])
    mask = K.sum(y_true,axis = [1,2,3]) + K.sum(y_pred,axis = [1,2,3])
    dice = (2*intersection + smooth) / (mask + smooth)
    return K.mean(dice)







