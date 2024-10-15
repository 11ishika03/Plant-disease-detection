# Plant-disease-detection
This project implements a deep learning model for the classification of plant diseases using TensorFlow and Keras. It consists of two main scripts: one for training the model and another for making predictions on new images.
## Introduction
Plant diseases can significantly impact agricultural productivity. This project aims to assist in the identification of various plant diseases by utilizing a convolutional neural network (CNN) to classify images of leaves. The model is trained on a dataset of plant leaf images, and can predict the disease category of a new leaf image.
### Requirements
Python 3.x
TensorFlow
NumPy
OpenCV

## Usage

### Training the Model

To train the model on the dataset, run the following command:

```bash
python train_model.py
```

This script will load the images from the `Data/PlantVillage` directory, preprocess them, and train the CNN model. After training, the model will be saved as `models/leaf_disease_model.h5`.

### Making Predictions

Once the model is trained, you can use it to predict the class of a new leaf image. Use the following command:

```bash
python predict.py <path-to-image>
```

Replace `<path-to-image>` with the path to the leaf image you want to classify. The script will output the predicted class number.

## Model Architecture

The model architecture consists of the following layers:

- 3 Convolutional layers followed by MaxPooling layers
- A Flatten layer
- A Dense layer with 128 neurons
- An output Dense layer with softmax activation to classify into 15 categories

The model is compiled using the Adam optimizer and categorical crossentropy loss function.

## License

This project is licensed under the MIT License.  
```

 
