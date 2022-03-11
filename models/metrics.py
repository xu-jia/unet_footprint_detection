from tensorflow.keras import backend as K
import tensorflow as tf

def iou_coef(y_true,y_pred,smooth=1e-6):
    y_pred = K.cast(K.argmax(y_pred, axis=-1),tf.float32)[...,tf.newaxis]
    intersection = K.sum(y_true*y_pred,axis = [1,2,3])
    union = K.sum(y_true+y_pred,axis = [1,2,3])-intersection
    iou = (intersection+smooth) / (union + smooth)
    return K.mean(iou)

def dice_coef(y_true, y_pred,  smooth=1e-6):
    y_pred = tf.argmax(y_pred, axis=-1)
    intersection = K.sum(y_true*y_pred,axis = [1,2,3])
    return (2. * intersection + smooth) / (K.sum(y_true,axis = [1,2,3]) + K.sum(y_pred,axis = [1,2,3]) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)







