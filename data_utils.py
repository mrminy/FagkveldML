import itertools
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy.misc import imresize
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

"""
Functions for fetching different datasets and for visualizing them
"""


def get_mnist(show_example=False):
    """
    Fetches the MNIST dataset of handwritten digital characters.
    :return:
    """
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if show_example:
        plot_image(x_train[0], title="MNIST example - " + str(y_train[0]))
    return x_train, x_test, y_train, y_test


def get_simple_classification_dataset(test_size=0.3, print_head=False):
    """
    Fetches a flower dataset.
    x = input features (four features per sample describing some of the flower's properties)
    y = correct labels (three classes of flowers)

    More information on the dataset: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html#sklearn.datasets.load_iris

    :param test_size: percentage size of the test set compared to the training set
    :param print_head: if true, print the first data sample
    :return: the splitted dataset in the order of: x_train, x_test, y_train, y_test
    """
    from sklearn.datasets import load_iris
    x, y = load_iris(return_X_y=True)
    if print_head:
        print(x[0], "-->", y[0])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    return x_train, x_test, y_train, y_test


def get_classification_dataset(test_size=0.3, print_head=False):
    """
    Binary classification problem where the features describe a digitized image of a fine needle aspirate (FNA) of a breast mass.

    More information on the dataset: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html#sklearn.datasets.load_breast_cancer
    Even more information: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

    :param test_size: percentage size of the test set compared to the training set
    :param print_head: if true, print the first data sample
    :return: the splitted dataset in the order of: x_train, x_test, y_train, y_test
    """
    from sklearn.datasets import load_breast_cancer
    x, y = load_breast_cancer(return_X_y=True)
    if print_head:
        print(x[0], "-->", y[0])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    return x_train, x_test, y_train, y_test


def plot_confusion_matrix(y_test, y_pred, classes=None, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Copied from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = confusion_matrix(y_test, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    if classes is not None:
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.show()


def plot_image(img_arr, title=None):
    plt.imshow(img_arr, cmap='gray')
    if title is not None:
        plt.title(title)
    plt.show()


def get_my_image(path, wanted_shape=(28, 28)):
    """
    Fetch a grayscale image on your local drive. Resize to a wanted shape is possible.
    :param path: path to image
    :param wanted_shape: if None, no resizing will be done. Otherwise, the image will be resized to the wanted shape
    :return: the local image as a numpy array
    """
    img = Image.open(path).convert('L')
    img.load()
    data = np.asarray(img, dtype="float32")
    if wanted_shape is not None:
        data = imresize(data, wanted_shape)
    return np.array([data], dtype="float32")


if __name__ == '__main__':
    # For testing purpose only...
    # get_simple_classification_dataset()
    # get_classification_dataset()
    # get_mnist()
    get_my_image('my_nine.png')
