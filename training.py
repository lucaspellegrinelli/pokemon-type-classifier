from model import SqueezeNet
from dataset import DatasetHandler
from defs import *

import os
import argparse

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from keras.optimizers import Adam

parser = argparse.ArgumentParser()
parser.add_argument("-device", action='store', dest='device', default="cpu", required=False)
parser.add_argument("--colab", action='store_true', dest='colab', default=False, required=False)
parser.add_argument("--model", action='store', dest='model', required=False)
args = parser.parse_args()

devices = {
  "cpu": "/device:CPU:0",
  "gpu": "/device:GPU:0",
  "xlagpu": "/device:XLA_GPU:0"
}

print("Parsed -device:", args.device, "=", devices[args.device])
print("Parsed --colab:", args.colab)
if args.model:
  print("Parsed --model", args.model)
print("\n")

hyperparameter = hyperparameter_defaults

with tf.device(devices[args.device]):
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
  print(df_dataset.drop("path", axis=1).sum().sort_values(ascending=False))

  # Creates the model
  if args.model:
    model = load_model(args.model)
  else:
    model = SqueezeNet(nb_classes=len(global_consts["types_label"]),
                       inputs=(configs["img_size"][0], configs["img_size"][1], 3))

    optimizer = Adam(learning_rate=hyperparameter["learning_rate"])
    model.compile(optimizer='adam', loss='binary_crossentropy')

  print(model.summary())

  if args.colab:
    base_path = "/content/gdrive/My Drive/"
  else:
    base_path = "models/"

  full_path = os.path.join(base_path, "pkm_model-{val_loss:.4f}.hdf5")
  callbacks = [
    ModelCheckpoint(full_path, monitor='val_loss', verbose=1, save_best_only=True),
    EarlyStopping(monitor='val_loss', patience=15)
  ]

  # Training time!
  history = model.fit_generator(
    generator=train_generator,
    steps_per_epoch=train_generator.n // train_generator.batch_size,
    validation_data=valid_generator,
    validation_steps=valid_generator.n // valid_generator.batch_size,
    epochs=hyperparameter["epochs"],
    verbose=1,
    callbacks=callbacks
  )
