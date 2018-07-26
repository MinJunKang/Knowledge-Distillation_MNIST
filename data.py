
# for mnist input data
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


class Dataset:
    def __init__(self, args):
        self.mnist = input_data.read_data_sets("./", one_hot=True)
        self.num_samples = self.mnist.train._num_examples
        self.batch_size = args.batch_size
        self.num_batches = int(self.num_samples / self.batch_size)
        print("Data Number : ",self.num_samples)

    def get_test_data(self):
        return (self.mnist.test.images, self.mnist.test.labels)

    def get_test_np(self):
        return (np.asarray(self.mnist.test.images), np.asarray(self.mnist.test.labels))

    def get_train_data(self):
        return self.mnist.train

    def get_train_np(self):
        return (np.asarray(self.mnist.train.images), np.asarray(self.mnist.train.labels))

    def get_validation_data(self):
        return (self.mnist.validation.images, self.mnist.validation.labels)

    def get_validation_np(self):
        return (np.asarray(self.mnist.validation.images), np.asarray(self.mnist.validation.labels))
