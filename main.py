import keras
from keras.models import Sequential
from keras.layers import Flatten, Conv2D, Dense, MaxPool2D, GlobalAveragePooling2D, Dropout
from keras import utils

import numpy as np
from PIL import Image
from helpers import NeptuneCallback, model_summary, load_Xy
from deepsense import neptune
ctx = neptune.Context()

base_path = "../input/train/"

model = Sequential()

model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPool2D())

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPool2D())

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D())

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D())

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D())

model.add(Conv2D(64, (1, 1), activation='relu'))
model.add(GlobalAveragePooling2D())
#model.add(Flatten())

#model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model_summary(model)

X_train, y_train, X_valid, y_valid = load_Xy(base_path)

print("\n")
print (len(X_valid))
print (len(y_valid))
X_valid = np.append(X_valid[::2], X_valid[1::2], axis=0)
y_valid = np.append(y_valid[::2], y_valid[1::2], axis=0)

Y_train = utils.to_categorical(y_train, 2)
Y_valid = utils.to_categorical(y_valid, 2)

print (len(X_train), len(Y_train), len(X_valid), len(Y_valid))
model.fit(X_train, Y_train,
      epochs=50,
      batch_size=32,
      validation_data=(X_valid, Y_valid),
      verbose=2,
      callbacks=[NeptuneCallback(X_valid, Y_valid, images_per_epoch=20)])
