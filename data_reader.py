import pandas as pd
import numpy as np
import cv2

from globals import image_size, batch_size


class Dataset:
    def __init__(self, file_name):
        self.filename = file_name
        self.labels = []
        self.images = []

        self.prepare_data()

    def prepare_data(self):
        data = pd.read_csv(self.filename)

        # Split the data and convert images to float 0 to 1
        self.labels = data.iloc[:, 0].values
        self.images = data.iloc[:, 1:].values.astype(np.float)
        self.images = np.multiply(self.images, 1 / 255.0)

    def display(self, index):
        image = self.images[index].reshape(image_size, image_size)
        print 'Label:', self.labels[index]
        cv2.imshow('', image)
        cv2.waitKey()
