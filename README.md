# Pokemon Type Classifier

A deep learning project that tries to predict a pokemon type based on images of it

## The dataset
This dataset was gathered by me using the Pokemon TCG cards and cropping the image roughly where the rectangle containing the pokemon image is in. With that said, this dataset is far from perfect since I didn't take into consideration that the rectangle where the image is in moved a little bit throughout the years and there are full cover cards with the images collected, so there are many poorly cropped images in the dataset, but that will do for this.

The .csv used to get the pokemon type from it's pokedex id was taken from [veekun's pokedex project](http://github.com/veekun/pokedex/).

### Examples

![](https://i.imgur.com/Y7AiT7L.jpg) ![](https://i.imgur.com/umSI8lZ.jpg)

## The model
For the project I decided to use the lightweight SqueezeNet architecture (arXiv 1602.07360) simply because the model is not that complicated to fit and the model file is small. The implementation of this model was taken from DT42's github (https://github.com/DT42/squeezenet_demo/) and modified a little to become a multi-label model.

## The results (from validation data)
This turned out to be alright, not the best results but I got really impressed by some of them. But to be fair, the task is actually pretty difficult since pokemons of each type doesn't have to share properties with each others and even if they do, with the amount of variety in each of the different cards make it a tough job. To be honest, some of the predictions looks like just follows the main color of the card. Some results are show as follows:

|            Original Pokemon          |       Input Image / Predictions      |
|:------------------------------------:|:------------------------------------:|
| ![](https://i.imgur.com/eh1yR1G.png) | ![](https://i.imgur.com/QL0qlz0.png) |
| ![](https://i.imgur.com/LLiF9Y0.png) | ![](https://i.imgur.com/WglhWVB.png) |
| ![](https://i.imgur.com/YOFYEqi.png) | ![](https://i.imgur.com/kSiSLaz.png) |
| ![](https://i.imgur.com/GxmeOLI.png) | ![](https://i.imgur.com/RAQDUC8.png) |
| ![](https://i.imgur.com/vBrk7p3.png) | ![](https://i.imgur.com/1ohSBbF.png) |
| ![](https://i.imgur.com/FsAYaVM.png) | ![](https://i.imgur.com/46V6N48.png) |
| ![](https://i.imgur.com/tTEokHj.png) | ![](https://i.imgur.com/Mi0YDYj.png) |
| ![](https://i.imgur.com/NTjgkGo.png) | ![](https://i.imgur.com/B3JYZI9.png) |
| ![](https://i.imgur.com/RM5RZOk.png) | ![](https://i.imgur.com/KwiB3ol.png) |
| ![](https://i.imgur.com/wuzrsYS.png) | ![](https://i.imgur.com/SZkvZY9.png) |
