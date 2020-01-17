from model import SqueezeNet
from dataset import DatasetHandler
from defs import *

import argparse

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

parser = argparse.ArgumentParser()
parser.add_argument("-device", action='store', dest='device', default="cpu", required=False)
parser.add_argument("--colab", action='store_true', dest='colab', default=False, required=False)
args = parser.parse_args()

devices = {
  "cpu": "/device:CPU:0",
  "gpu": "/device:GPU:0",
  "xlagpu": "/device:XLA_GPU:0"
}

print("Parsed -device:", args.device, "=", devices[args.device])
print("Parsed --colab", args.colab)

with tf.device(devices[args.device]):
  # Creates the dataset
  image_path = "dataset/images/"
  types_csv_path = "dataset/pokemon_types.csv"
  data_gen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
  )

  ds_handler = DatasetHandler(image_path, types_csv_path, data_gen)
  df_dataset, train_generator, valid_generator = ds_handler.create_dataset(verbose=True)

  # Displays the number of images of each type
  # water       1623
  # normal      1311
  # flying      1300
  # psychic     1288
  # grass       1027
  # fire        1017
  # poison       879
  # ground       865
  # electric     844
  # bug          741
  # dragon       740
  # steel        707
  # rock         666
  # dark         652
  # fighting     626
  # fairy        523
  # ghost        488
  # ice          370
  print("Number of images of each type:")
  df_dataset.drop("path", axis=1).sum().sort_values(ascending=False)

  # Creates the model
  model = SqueezeNet(nb_classes=len(global_consts["types_label"]),
                     inputs=(configs["img_size"][0], configs["img_size"][1], 3))

  print(model.summary())

  # Compiles the model
  model.compile(optimizer='adam', loss='binary_crossentropy')

  base_path = "/content/gdrive/My Drive/" if args.colab else ""
  callbacks = [
    ModelCheckpoint(base_path + "models/pkm_model-{val_loss:.4f}.hdf5",
                    monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
  ]

  # Training time!
  history = model.fit_generator(
    generator=train_generator,
    steps_per_epoch=train_generator.n // train_generator.batch_size,
    validation_data=valid_generator,
    validation_steps=valid_generator.n // valid_generator.batch_size,
    epochs=configs["epochs"],
    verbose=1,
    callbacks=callbacks
  )
