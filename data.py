# for mnist input data
from tensorflow.examples.tutorials.mnist import input_data
import Header

# For data read
class Dataset:
    def __init__(self):
        self.mnist = input_data.read_data_sets("./", one_hot = True)
        print("Train Data Number : ",self.mnist.train._num_examples)
        print("Validation Data Number : ",self.mnist.validation._num_examples)
        print("Test Data Number : ",self.mnist.test._num_examples)

    def get_test_data(self):
        return (self.mnist.test.images, self.mnist.test.labels)

    # Convert to numpy type
    def get_test_np(self):
        return (Header.np.asarray(self.mnist.test.images), Header.np.asarray(self.mnist.test.labels))

    def get_train_data(self):
        return self.mnist.train

    # Convert to numpy type
    def get_train_np(self):
        return (Header.np.asarray(self.mnist.train.images), Header.np.asarray(self.mnist.train.labels))

    def get_validation_data(self):
        return (self.mnist.validation.images, self.mnist.validation.labels)

    # Convert to numpy type
    def get_validation_np(self):
        return (Header.np.asarray(self.mnist.validation.images), Header.np.asarray(self.mnist.validation.labels))
