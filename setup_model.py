# Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.utils.np_utils import to_categorical
from get_kaggle_data import get_kaggle_mnist


# Read data

def vectorize_img_data(data):
    dim1 = data.shape[0]
    dim2 = data.shape[1] * data.shape[2]

    data_rsp = data.reshape((dim1, dim2))

    return data_rsp


def standardize_images(images, pixels=255):
    return images / pixels


def setup_data():
    train = pd.read_csv("train.csv")
    test_images = pd.read_csv("test.csv").values.astype('float32')

    # Pre processing
    train_images = train.ix[:, 1:].values.astype('float32')
    train_labels = train.ix[:, 0].values.astype('int32')

    train_images = train_images.reshape(train_images.shape[0], 28, 28)

    train_features_raw = vectorize_img_data(train_images)
    x_train = standardize_images(train_features_raw)
    x_test = standardize_images(test_images)

    y_train = to_categorical(train_labels)

    return x_train, x_test, y_train


def print_images(images, labels, numbers=range(6, 9)):
    for i in numbers:
        plt.subplot(330 + (i + 1))
        plt.imshow(images[i], cmap=plt.get_cmap('gray'))
        plt.title(labels[i])


def run_neural_net(features, target):
    num_features = features.shape[1]
    mdl = Sequential()
    mdl.add(Dense(32, activation='relu', input_dim=num_features))
    mdl.add(Dense(16, activation='relu'))
    mdl.add(Dense(10, activation='softmax'))

    mdl.compile(optimizer=RMSprop(lr=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    hist = mdl.fit(
        features,
        target,
        validation_split=0.05,
        nb_epoch=10,
        batch_size=64
    )

    return mdl, hist


# Design neural network

# fix random seed for reproducibility
seed = 43
np.random.seed(seed)

print("Train input shape: {shape}".format(shape=train_images.shape))
print("Train output shape: {shape}".format(shape=train_labels.shape))

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=(28 * 28)))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile network

model.compile(
    optimizer=RMSprop(lr=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_images,
    train_labels,
    validation_split=0.05,
    nb_epoch=10,
    batch_size=64
)

# Predict
predictions = model.predict_classes(test_images, verbose=0)

submissions = pd.DataFrame({"ImageId": list(range(1, len(predictions) + 1)),
                            "Label": predictions})

# Script

get_kaggle_mnist()
x_train, x_test, y_train = setup_data()

nn, outpt = run_neural_net(features=x_train, target=y_train)
predicts = nn.predict_classes(x_te, verbose=0)