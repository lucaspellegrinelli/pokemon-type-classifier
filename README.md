# Pokemon Type Classifier

A deep learning project that tries to predict a pokemon type based on images of it

## The dataset
This dataset was gathered by me using the Pokemon TCG cards and cropping the image roughly where the rectangle containing the pokemon image is in. With that said, this dataset is far from perfect since I didn't take into consideration that the rectangle where the image is in moved a little bit throughout the years and there are full cover cards with the images collected, so there are many poorly cropped images in the dataset, but that will do for this.

The .csv used to get the pokemon type from it's pokedex id was taken from [veekun's pokedex project](http://github.com/veekun/pokedex/).

### Examples

![](https://i.imgur.com/Y7AiT7L.jpg) ![](https://i.imgur.com/umSI8lZ.jpg)

## The model
Using the Google's InceptionV3 pretrained model as a base model, I trained a fine tuned version of it to tackle the project.

## The results
TODO
