import tensorflow.keras as keras

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.utils import to_categorical
import helper_api

num_classes = 8
batch_size = 5
epochs = 3

print("Downloading dataset...")
helper_api.download(
    url='http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar')

print("Loading dataset...")
raw_X, Y, mean_w, mean_h = helper_api.loadDataset(labels=num_classes)
X = np.asarray(helper_api.standartize(raw_X, mean_w, mean_h))
print(X.shape)
print(np.asarray(Y).shape)

# normalize pics, categorize labels and split data
X = X / 255.0
Y = to_categorical(Y, num_classes)

(train_X, test_X, train_Y, test_Y) = train_test_split(
    X, Y, test_size=0.35, random_state=8888)

# augmentation model
aug_generator = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    vertical_flip=False)
aug_generator.fit(train_X)

# see augmented imgs
#batches = 0
# for x_batch, y_batch in aug_generator.flow(train_X, train_Y, batch_size=30):
#    batches += 1
#    helper_api.showHead(x_batch[0:12], x_batch[13:25], 2)
#    if batches >= 3:
#        break

# model initialization
model = Sequential([
    Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), activation='relu'),
    BatchNormalization(),
    MaxPool2D(pool_size=2),

    Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation='relu'),
    BatchNormalization(),
    MaxPool2D(pool_size=2),

    Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu'),
    BatchNormalization(),
    MaxPool2D(pool_size=2),

    Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu'),
    BatchNormalization(),
    MaxPool2D(pool_size=2),

    Flatten(),
    Dense(units=num_classes, activation='softmax'),
])

loss = keras.losses.CategoricalCrossentropy()
opt = keras.optimizers.Adam(lr=0.000001)
best_model = ModelCheckpoint(
    'dogs-iter1model.h5', save_best_only=True, verbose=1)
model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
H = model.fit_generator(aug_generator.flow(train_X, train_Y, batch_size=batch_size),
                        epochs=epochs, validation_data=(test_X, test_Y), callbacks=[best_model])

# testing
test_loss, test_acc = model.evaluate(test_X, test_Y)
print('test_acc:', test_acc)

# plot
epochs = range(1, len(H.history['acc']) + 1)

plt.figure(1, figsize=(8, 5))
plt.title("Training and test accuracy")
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.plot(epochs, H.history['acc'], 'r', label='train')
plt.plot(epochs, H.history['val_acc'], 'b', label='test')
plt.legend()
plt.show()
plt.clf()
