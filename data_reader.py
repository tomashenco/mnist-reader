import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from math import floor
import cv2

from globals import image_size, batch_size, num_classes


class TrainDataset:
    def __init__(self, file_name):
        self.filename = file_name
        data = pd.read_csv(self.filename)

        # Split the data and convert images to float 0 to 1
        labels = data.iloc[:, 0].values

        labels_one_hot = self.convert_to_one_hot(labels)
        images = data.iloc[:, 1:].values.astype(np.float)
        images = np.multiply(images, 1 / 255.0)

        x, y = shuffle(images, labels_one_hot)
        self.images_train = x[:-2000]
        self.images_val = x[-2000:]
        self.labels_train = y[:-2000]
        self.labels_val = y[-2000:]

    @staticmethod
    def convert_to_one_hot(labels):
        num_samples = labels.shape[0]
        labels_one_hot = np.zeros((num_samples, num_classes))
        labels_one_hot[np.arange(num_samples), labels] = 1
        return labels_one_hot

    def display(self, index):
        image = self.images_val[index].reshape(image_size, image_size)
        cv2.imshow('', image)
        cv2.waitKey()

    def iterate_batches(self):
        n_batches = int(floor(self.images_train.shape[0] / batch_size))

        x, y = shuffle(self.images_train, self.labels_train)
        for n in xrange(n_batches):
            x_batch = x[n*batch_size:(n+1)*batch_size]
            y_batch = y[n*batch_size:(n+1)*batch_size]

            yield x_batch, y_batch


class TestDataset:
    def __init__(self, file_name):
        self.filename = file_name
        images = pd.read_csv(self.filename).values
        self.images_train = np.multiply(images, 1/255.0).astype(np.float32)

    def display(self, index):
        image = self.images_train[index].reshape(image_size, image_size)
        cv2.imshow('', image)
        cv2.waitKey()


test = TrainDataset('train.csv')
