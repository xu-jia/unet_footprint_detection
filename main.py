from data_load import Generator
from models import u_net
import tensorflow as tf
if __name__ == '__main__':
    model = u_net.build_model(
            base_model=None,
            input_height=512,
            input_width=512,
            n_channels=3,
            n_filters=64,
            n_classes=2,
            dropout_prob=0.2,
            encoder_freeze=False)
tf.keras.utils.plot_model(model,to_file='model.png', show_shapes=True)




