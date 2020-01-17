from dataset import DatasetHandler

import argparse

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

parser = argparse.ArgumentParser()
parser.add_argument("-device", action='store', dest='device', default="cpu", required=False)
parser.add_argument("-model", action='store', dest='model', required=True)
args = parser.parse_args()

devices = {
  "cpu": "/device:CPU:0",
  "gpu": "/device:GPU:0",
  "xlagpu": "/device:XLA_GPU:0"
}

print("Parsed -device:", args.device, "=", devices[args.device])
print("Parsed --model", args.model)
print("\n")

ds_handler = DatasetHandler(image_path, types_csv_path, data_gen)
df_dataset, train_generator, valid_generator = ds_handler.create_dataset(verbose=True)

model = load_model(args.model)

x_batch, y_batch = valid_generator.next()
for image in x_batch:
  plt.imshow(image)
  plt.show()
  prediction = model.predict(np.expand_dims(image, axis=0))[0]
  prediction_labeled = sorted(list(zip(prediction, types_label)), reverse=True)
  for prob, name in prediction_labeled:
    print(name, prob * 100, "%")
