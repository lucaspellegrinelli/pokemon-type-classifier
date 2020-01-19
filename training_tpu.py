import tensorflow as tf
import os

from model import SqueezeNet
from dataset import DatasetHandler
from defs import *

from keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

try:
  device_name = os.environ["COLAB_TPU_ADDR"]
  TPU_ADDRESS = "grpc://" + device_name
  print("Found TPU at: {}".format(TPU_ADDRESS))
except KeyError:
  print("TPU not found")
  exit()

# Creates the dataset
image_path = "dataset/images/"
types_csv_path = "dataset/pokemon_types.csv"
data_gen = ImageDataGenerator(
  rescale=1.0 / 255.0,
  validation_split=configs["val_split"],
  zoom_range=configs["zoom_range"],
  horizontal_flip=True,
  fill_mode="nearest",
  rotation_range=configs["rotation_range"]
)

print("Gathering data")
ds_handler = DatasetHandler(image_path, types_csv_path, data_gen)
df_dataset, train_generator, valid_generator = ds_handler.create_dataset(verbose=True)
traindataset = tf.data.Dataset.from_generator(generator=train_generator,
                                               output_types=(tf.int32),
                                               output_shapes=(17,))
print(df_dataset.head())


TPU_WORKER = 'grpc://' + os.environ['COLAB_TPU_ADDR']

tf.keras.backend.clear_session()

resolver = tf.contrib.cluster_resolver.TPUClusterResolver(TPU_WORKER)
tf.contrib.distribute.initialize_tpu_system(resolver)
strategy = tf.contrib.distribute.TPUStrategy(resolver)

with strategy.scope():
  model = SqueezeNet(nb_classes=len(global_consts["types_label"]),
          inputs=(configs["img_size"][0], configs["img_size"][1], 3))

  opt = tf.train.AdamOptimizer(configs["learning_rate"])
  model.compile(optimizer=opt, loss='binary_crossentropy')

# Training time!
history = tpu_model.fit(
  generator=traindataset,
  epochs=configs["epochs"],
  verbose=1
)
