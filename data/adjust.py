# script for adjusting the original images for use in the cnn

import os
import cv2

# parameters that can be adjusted

WIDTH = 256
HEIGHT = 256
USE_GRAYSCALE = False

# script

IMAGE_SHAPE = (256,256) if USE_GRAYSCALE else (256,256,3)
CHANNELS = cv2.IMREAD_GRAYSCALE if USE_GRAYSCALE else cv2.IMREAD_COLOR

subdirectories = ['berk', 'malik', 'papa', 'mama', 'oma']

for subdirectory in subdirectories:

    directory = os.path.join('original', subdirectory)
    if not os.path.exists(directory):
        continue

    counter = 0

    for filename in os.listdir(directory):

        # load old img

        imagepath = os.path.join(directory, filename)
        img = cv2.imread(imagepath, CHANNELS)

        # resize

        resized_img = cv2.resize(img, (WIDTH, HEIGHT))

        # save new img

        new_path = os.path.join('./train', subdirectory, f'{str(counter)}.jpg')
        cv2.imwrite(new_path, resized_img)
        print(new_path)
        counter = counter + 1