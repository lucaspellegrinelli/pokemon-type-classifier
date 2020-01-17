from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

from defs import *

class ModelCreator:
  def __init__(self):
    pass

  @staticmethod
  def create_model():
    # Creates the InceptionV3 model as a base model
    base_model = InceptionV3(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(len(global_consts["types_label"]), activation='sigmoid')(x)

    # Joins the InceptionV3 with the latter layers
    return Model(inputs=base_model.input, outputs=x)
