# Pokemon Type Classifier

A deep learning project that tries to predict a pokemon type based on images of it

## The dataset
This dataset was gathered by me using the Pokemon TCG cards and cropping the image roughly where the rectangle containing the pokemon image is in. With that said, this dataset is far from perfect since I didn't take into consideration that the rectangle where the image is in moved a little bit throughout the years and there are full cover cards with the images collected, so there are many poorly cropped images in the dataset, but that will do for this.

The .csv used to get the pokemon type from it's pokedex id was taken from [veekun's pokedex project](http://github.com/veekun/pokedex/).

### Examples

![](https://i.imgur.com/Y7AiT7L.jpg) ![](https://i.imgur.com/umSI8lZ.jpg)

## The model
For the project I decided to use the lightweight SqueezeNet architecture (arXiv 1602.07360) simply because the model is not that complicated to fit and the model file is small. The implementation of this model was taken from DT42's github (https://github.com/DT42/squeezenet_demo/) and modified a little to become a multi-label model.

## The results
TODO
