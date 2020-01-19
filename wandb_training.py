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

import wandb
from wandb.keras import WandbCallback
wandb.init(config=configs, project="pokemon-type-classifier")
hyperparameter = wandb.config

with tf.device("/device:GPU:0"):
  # Creates the dataset
  image_path = "dataset/images/"
  types_csv_path = "dataset/pokemon_types.csv"
  data_gen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=hyperparameter["val_split"],
    zoom_range=hyperparameter["zoom_range"],
    horizontal_flip=True,
    fill_mode="nearest",
    rotation_range=hyperparameter["rotation_range"]
  )

  ds_handler = DatasetHandler(image_path, types_csv_path, data_gen)
  df_dataset, train_generator, valid_generator = ds_handler.create_dataset(verbose=True)

  # Creates the model
  model = SqueezeNet(nb_classes=len(global_consts["types_label"]),
                     inputs=(hyperparameter["img_size"][0], hyperparameter["img_size"][1], 3))
  optimizer = Adam(lr=hyperparameter["learning_rate"])
  model.compile(optimizer='adam', loss='binary_crossentropy')

  full_path = os.path.join(wandb.run.dir, "pkm_model-{val_loss:.4f}.hdf5")
  callbacks = [
    ModelCheckpoint(full_path, monitor='val_loss', verbose=1, save_best_only=True),
    EarlyStopping(monitor='val_loss', patience=15),
    WandbCallback()
  ]

  # Training time!
  history = model.fit_generator(
    generator=train_generator,
    steps_per_epoch=train_generator.n // train_generator.batch_size,
    validation_data=valid_generator,
    validation_steps=valid_generator.n // valid_generator.batch_size,
    epochs=hyperparameter["epochs"],
    verbose=2,
    callbacks=callbacks
  )
