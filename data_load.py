import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img,img_to_array
import numpy as np
import pandas as pd
AUTOTUNE = tf.data.AUTOTUNE
def Generator(input_img_paths,target_img_paths,batch_size,train=False):
    # create dataset from slices of filenames
    dataset = tf.data.Dataset.from_tensor_slices((input_img_paths, target_img_paths))
    # parse the files to pixels values
    dataset = dataset.map(tf.autograph.experimental.do_not_convert(parse_function), num_parallel_calls=AUTOTUNE)
    if train:
        dataset = dataset.cache()
        dataset = dataset.shuffle(buffer_size=500)
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(tf.autograph.experimental.do_not_convert(Augment()), num_parallel_calls=AUTOTUNE)
        return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    else:
        return dataset.batch(batch_size)

# parse function: image file to pixel values
def parse_function(path_img, path_target):
    img_string = tf.io.read_file(path_img)
    target_string = tf.io.read_file(path_target)
    img = tf.io.decode_png(img_string, channels=3)
    img = tf.cast(img, tf.float32) / 255.0
    target = tf.io.decode_png(target_string, channels=1)
    return img,target

class Augment(tf.keras.layers.Layer):
  def __init__(self, seed=42):
    super().__init__()
    # both use the same seed, so they'll make the same random changes.
    self.augment_inputs  = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical",seed=seed),
            tf.keras.layers.RandomRotation(0.2,seed=seed),
            tf.keras.layers.RandomContrast(0.2, seed=seed)
        ])
    self.augment_labels = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical",seed=seed),
            tf.keras.layers.RandomRotation(0.2,seed=seed)
        ])
  def call(self, img, target):
    img = self.augment_inputs(img)
    target = self.augment_labels(target)
    return img, target
class Datagenerator(tf.keras.utils.Sequence):
    def __init__(self, batch_size, input_height, input_width , df_file, image_paths, target_paths,train=False):
        df = pd.read_csv(df_file)
        self.batch_size = batch_size
        self.img_size =  (input_height, input_width)
        self.input_img_paths = [image_paths+ im_id for im_id in df["ImageId"]]
        self.target_img_paths = [target_paths + im_id for im_id in df["ImageId"]]
        self.train = train
        self.on_epoch_end()

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """batch data generation"""
        i = idx * self.batch_size
        batch_indexes = self.indexes[i: i + self.batch_size]
        """generate data tuple (input, target) correspond to batch_indexes."""
        batch_input_img_paths = [self.input_img_paths[id] for id in batch_indexes]
        batch_target_img_paths = [self.target_img_paths[id] for id in batch_indexes]
        # input image
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        # target image
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")

        for j, path_img, path_target in zip(range(len(batch_indexes)), batch_input_img_paths, batch_target_img_paths):
            img = load_img(path_img, target_size=self.img_size)
            target = load_img(path_target, target_size=self.img_size, color_mode="grayscale")
            img = img_to_array(img)
            target = img_to_array(target)
            # data augmentation when we generate training data
            if self.train:
                img, target = self.__data_augmentation(img, target)
            x[j] = img / 255.0
            # y[j] = np.expand_dims(target, 2)
        return x, y


    def __data_augmentation(self, img, target):
        """data augmentation"""
        img = tf.image.random_flip_left_right(img,seed=40)
        target = tf.image.random_flip_left_right(target,seed=40)
        img = tf.image.random_flip_up_down(img,seed=40)
        target = tf.image.random_flip_up_down(target, seed=40)
        img = tf.image.random_brightness(img, 0.2)
        img = tf.image.random_contrast(img, 0.5, 2.0)
        img = tf.image.random_saturation(img, 0.75, 1.25)
        return img,target

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.input_img_paths))
        # shuffle in the end of epoch when we generate the training data
        if self.train:
            np.random.shuffle(self.indexes)


