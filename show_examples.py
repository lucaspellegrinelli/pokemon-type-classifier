from dataset import DatasetHandler

import argparse

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

parser = argparse.ArgumentParser()
parser.add_argument("-device", action='store', dest='device', default="cpu", required=False)
parser.add_argument("-model", action='store', dest='model', required=True)
parser.add_argument("-numexamples", type=int, action='store', dest='numexamples', default=8, required=False)
args = parser.parse_args()

devices = {
  "cpu": "/device:CPU:0",
  "gpu": "/device:GPU:0",
  "xlagpu": "/device:XLA_GPU:0"
}

print("Parsed -device:", args.device, "=", devices[args.device])
print("Parsed -model", args.model)
print("Parsed -numexamples", args.numexamples)
print("\n")

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

model = load_model(args.model)

example_count = 0

while example_count < args.numexamples:
  x_batch, y_batch = valid_generator.next()
  for i, image in enumerate(x_batch):
    example_count += 1
    if example_count > args.numexamples:
      break
    plt.imshow(image)
    plt.show()
    prediction = model.predict(np.expand_dims(image, axis=0))[0]
    prediction_labeled = sorted(list(zip(prediction, global_consts["types_label"])), reverse=True)
    for prob, name in prediction_labeled:
      print(name, prob * 100, "%")
