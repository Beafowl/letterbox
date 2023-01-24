import os
import sys
import tensorflow as tf
from tensorflow import keras
from keras import layers
import keras
import numpy as np
import cv2
import matplotlib.pyplot as plt

# parameters that can be adjusted

WIDTH = 256
HEIGHT = 256
USE_GRAYSCALE = False
IMAGES_PER_CLASS = 30
CLASS_AMOUNT = 5

TEST_IMAGES_AMOUNT = 4

# script

IMAGE_SHAPE = (WIDTH,HEIGHT) if USE_GRAYSCALE else (WIDTH,HEIGHT,3)
CHANNELS = cv2.IMREAD_GRAYSCALE if USE_GRAYSCALE else cv2.IMREAD_COLOR

# input: pixel values, image with classifications (berk, oma, malik, papa, mama)

class_names = np.array(['berk', 'oma', 'malik', 'papa', 'mama'])

def help():
        print("cnn.py --create | Create model with the data specified in data/train and save it")
        print("cnn.py --predict | Predict with the saved model with the data in data/test")

def create_model():


        # before training, every image needs to be labeled with a class (TODO)
        # in this case, we need to consider the dimensions of the images for the microcontroller (resolution, rgb vs grayscale etc.)

        # load images and adjust dimension (TODO)

        train_images = np.zeros(shape=(class_names.size, IMAGES_PER_CLASS, WIDTH, HEIGHT, 3), dtype='uint8')
        train_labels = np.zeros(shape=(class_names.size, IMAGES_PER_CLASS, class_names.size), dtype='uint8')

        directory = '..\\data\\train'

        for class_index in range(0, class_names.size):

                # load old img

                imagefolder = os.path.join(directory, class_names[class_index])

                for i in range(0, IMAGES_PER_CLASS):

                        imagepath = os.path.join(imagefolder, f'{i}.jpg')
                        if not os.path.exists(imagepath):
                                continue
                        img = cv2.imread(imagepath, CHANNELS)

                        # TODO: Normalize images

                        img = img / 256
                        #cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX)

                        train_images[(class_index, i)] = img
                        train_labels[(class_index, i)] = np.eye(class_names.size)[class_index]

        # train_images has dimension (30, 256, 256, 3)
        # reshape?

        train_images = tf.reshape(train_images, [WIDTH, HEIGHT, 3])
        train_labels = tf.reshape(train_labels, [class_names.size])

        # create model
        model = keras.models.Sequential()

        model.add(layers.Input((WIDTH, HEIGHT, 3)))

        model.add(layers.Conv2D(64, (3,3))) # adjust input shape when images are ready and library on the microcontroller is set
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D(pool_size=(2,2)))

        model.add(layers.Conv2D(127, (3,3))) # adjust input shape when images are ready and library on the microcontroller is set
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D(pool_size=(2,2)))

        model.add(layers.Flatten())
        model.add(layers.Dense(64))

        model.add(layers.Activation('relu'))

        model.add(layers.Dense(class_names.size))
        model.add(layers.Activation('sigmoid'))

        loss = keras.losses.CategoricalCrossentropy()
        optim = keras.optimizers.Adam(learning_rate=0.001)
        metrics = ['accuracy']

        model.compile(optimizer=optim, loss=loss, metrics=metrics)

        # train the model

        batch_size = 1
        epochs = 1

        print(model.summary())

        model.fit(train_images[class_index], train_labels[class_index], epochs=epochs, batch_size=batch_size, verbose=2, validation_split=0.1)

        # check if the model accepts an image

        try:

                prediction = model.predict(train_images[(0,0)])
                print(prediction)

        except Exception as e:
                print(e)

        else:
                # save the model
                model.save("nn")
                #model.save_weights("nn_weights")

                # json_string = model.to_json()
def predict():

        test_images = np.zeros(shape=(class_names.size, TEST_IMAGES_AMOUNT, WIDTH, HEIGHT, 3), dtype='uint8')
        test_labels = np.zeros(shape=(class_names.size, TEST_IMAGES_AMOUNT, class_names.size), dtype='uint8')

        # something weird happens at i = 4
        for class_index in range(0, class_names.size-1):

                # load old img

                directory = '..\\data\\test'
                batch_size = 5

                imagefolder = os.path.join(directory, class_names[class_index])

                
                for i in range(0, TEST_IMAGES_AMOUNT):

                        imagepath = os.path.join(imagefolder, f'{i}.jpg')
                        if not os.path.exists(imagepath):
                                continue
                        img = cv2.imread(imagepath, CHANNELS)

                        #cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX)

                        #print(img.shape)                        
                        #print(i)
                        #print(class_index)
                        #print(test_images[(class_index, i)].shape)


                        test_images[(class_index, i)] = img
                        test_labels[(class_index, i)] = np.eye(class_names.size)[class_index]

                        # TODO: image shows black, and shape seems to be (32, 256) here but it should be (256, 256)

                        #cv2.imshow('xd', test_images[(class_index, i)])
                        #cv2.waitKey(0)
                        #print(len(test_images[3]))

        model = keras.models.load_model('nn')

        print(model.summary())

        #for class_index in range(0, class_names.size):
        #        model.evaluate(test_images[class_index], test_labels[class_index], batch_size=batch_size, verbose=2)
        
        #print(test_images[(3,0)].shape)

        #cv2.imshow("xd", test_images[(3,0)])
        #cv2.waitKey(0)

        print(test_images[(3,1)].shape)
        cv2.imshow('xd', test_images[(3,1)])
        cv2.waitKey(0)

        prediction = model.predict(test_images[(3,1)])
        print(prediction)

if __name__ == '__main__':

        if len(sys.argv) != 2:
                help()
                exit()

        if sys.argv[1] == '--create':
                create_model()
        elif sys.argv[1] == '--predict':
                predict()
        else:
                print("cnn.py --create | Create model with the data specified in data/train and save it")
                print("cnn.py --predict | Predict with the saved model with the data in data/test")
