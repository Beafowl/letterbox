import numpy as np
import tensorflow as tf
from keras import layers
import keras

model = keras.models.Sequential()

model.add(layers.Input(1))
model.add(layers.Dense(units=1, activation=None))

loss = keras.losses.CategoricalCrossentropy(from_logits=True)
optim = keras.optimizers.Adam(learning_rate=0.1)
metrics = ['accuracy']

model.compile(optimizer=optim, loss=loss, metrics=metrics)

print(model.summary())

x = [1, 3]
y = [2, 3]

epochs = 5
batch_size = 1000

model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=2)

print(model.predict([3]))