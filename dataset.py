from defs import *

import pandas as pd
import random
from imutils import paths
import numpy as np

class DatasetHandler:
  def __init__(self, image_paths, types_csv, datagen):
    # Setting seed for reproducibility
    random.seed(42);

    # Load and shuffle image paths
    self.image_paths = list(paths.list_images(image_paths))
    random.shuffle(self.image_paths)

    # Load types csv
    self.types = pd.read_csv(types_csv)

    # ImageGenerator options
    self.datagen = datagen

  # With the pokemon pokedex id as input, this function returns in an
  # one-hot-encoding form their encoded types
  def pokemons_to_labels(self):
    types_by_pokemon = {}
    min_id = self.types.pokemon_id.min()
    max_id = self.types[self.types.pokemon_id <= 1000].pokemon_id.max()

    for i in range(min_id, max_id + 1):
      types_id = self.types[self.types.pokemon_id == i]["type_id"].to_numpy()
      one_hot = [0] * len(global_consts["types_label"])
      for t_id in types_id:
        one_hot[t_id - 1] = 1

      types_by_pokemon[i] = one_hot

    return types_by_pokemon

  # Creates a DataFrame with all pokemon image paths and their correspondent
  # one-hot-encoded types.
  def create_dataset(self, verbose=False):
    types_by_pokemon = self.pokemons_to_labels()

    paths, labels = [], []
    for path in self.image_paths:
      pkm_id = int(path.split("/")[-1].split("-")[0])
      labels.append(types_by_pokemon[pkm_id])
      paths.append(path)

    dataset_df = pd.DataFrame(labels, columns=global_consts["types_label"])
    dataset_df["path"] = paths

    train_gen, val_gen = self.create_data_generators(dataset_df)
    return dataset_df, train_gen, val_gen

  # Creates the data generators from the dataframe inputted
  def create_data_generators(self, df_dataset):
    dataset_cols = ["path"] + global_consts["types_label"]

    for col in dataset_cols[1:]:
      df_dataset[col] = pd.to_numeric(df_dataset[col])

    # Training data generator
    train_generator = self.datagen.flow_from_dataframe(
      dataframe=df_dataset,
      x_col=dataset_cols[0],
      y_col=dataset_cols[1:],
      subset="training",
      batch_size=configs["batch_size"],
      shuffle=True,
      class_mode="other",
      target_size=configs["img_size"],
      seed=42
    )

    # Validation data generator
    valid_generator = self.datagen.flow_from_dataframe(
      dataframe=df_dataset,
      x_col=dataset_cols[0],
      y_col=dataset_cols[1:],
      subset="validation",
      batch_size=configs["batch_size"],
      shuffle=True,
      class_mode="other",
      target_size=configs["img_size"],
      seed=42
    )

    return train_generator, valid_generator
