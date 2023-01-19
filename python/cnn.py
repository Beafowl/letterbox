import os

import tensorflow as tf
import matplotlib.pyplot as plt

# input: pixel values, image with classifications (berk, oma, malik, papa, mama)

class_names = ['berk', 'oma', 'malik', 'papa', 'mama']

# before training, every image needs to be labeled with a class (TODO)
# in this case, we need to consider the dimensions of the images for the microcontroller (resolution, rgb vs grayscale etc.)

# load images and adjust dimension (TODO)

train_images = ...
train_labels = ...

test_images = ...
test_labels = ...

# create model
model = keras.models.Sequential()
model.add(layers.Conv2D(32, (3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(32,32,3))) # adjust input shape when images are ready and library on the microcontroller is set

print(model)

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = keras.optimizers.Adam(lr=0.001)
metrics = ['accuracy']

model.compile(optimizer=optim, loss=loss, metrics=metrics)

# train the model

batch_size = 64
epochs = 5



#model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, verbose=2)

# evaluate

#model.evaluate(test_images, test_labels, batch_size=batch_size, verbose=2)