#Modules

import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical

from keras.models import Sequential
from keras.layers import Dense , Dropout
from keras.optimizers import RMSprop


#Get data
train = pd.read_csv("train.csv")
test_images = pd.read_csv("test.csv").values.astype('float32')

#Pre processing
train_images = (train.ix[:,1:].values).astype('float32')
train_labels = train.ix[:,0].values.astype('int32')

train_images = train_images.reshape(train_images.shape[0],  28, 28)

for i in range(6, 9):
    plt.subplot(330 + (i+1))
    plt.imshow(train_images[i], cmap=plt.get_cmap('gray'))
    plt.title(train_labels[i]);

train_images = train_images.reshape((42000, 28 * 28))

train_images = train_images / 255
test_images = test_images / 255


train_labels = to_categorical(train_labels)
num_classes = train_labels.shape[1]

#Design neural network

# fix random seed for reproducibility
seed = 43
np.random.seed(seed)

print("Train input shape: {shape}".format(shape=train_images.shape))
print("Train output shape: {shape}".format(shape=train_labels.shape))

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=(28 * 28)))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))

#Compile network

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

#Predict
predictions = model.predict_classes(test_images, verbose=0)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})

