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
  def pokemon_to_label(self, pokemon_id):
    pkm_types = np.array(self.types[self.types.pokemon_id == pokemon_id]["type_id"])
    one_hot = [0] * len(global_consts["types_label"])
    for t in pkm_types:
      one_hot[t - 1] = 1

    return one_hot

  # Creates a DataFrame with all pokemon image paths and their correspondent
  # one-hot-encoded types.
  def create_dataset(self, verbose=False):
    dataset_cols = ["path"] + global_consts["types_label"]
    dataset_df = pd.DataFrame(columns=dataset_cols)

    for i, path in enumerate(self.image_paths):
      if i % 1000 == 0 and verbose:
        print("Loading images", str(i) + "/" + str(len(self.image_paths)))

      pkm_id = int(path.split("/")[-1].split("-")[0])
      pkm_entry = {"path": path}
      for label, col in zip(self.pokemon_to_label(pkm_id), dataset_cols[1:]):
        pkm_entry[col] = label

      dataset_df = dataset_df.append(pkm_entry, ignore_index=True)

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
