from time import gmtime, strftime

import numpy as np
from keras import callbacks
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical

from data_utils import get_mnist, get_my_image, plot_image

"""
Script for training and testing deep learning models on the MNIST dataset
"""


def preprocess_dataset(x_train, x_test, y_train, y_test, num_classes=10):
    """
    Preprocess the dataset for deep learning model.
    1. Make sure the dataset has the correct shape (784x1)
    2. Convert it to type float32 for more efficient computation
    3. Divide pixel intensity values on 255 to scale the input features to [0,1]
    4. Make the labels (classes) into a one-hot array for each sample. For example: 4 = [0,0,0,0,1,0,0,0,0,0]
    """
    # Reshaping input features to 784x1 from 28x28
    x_train = x_train.reshape(60000, img_cols * img_rows, )
    x_test = x_test.reshape(10000, img_cols * img_rows, )

    # Casting to float32 for efficiency
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # Rescaling pixel intensity values
    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Converting labels into one-hot arrays
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    return x_train, x_test, y_train, y_test


def build_model(input_shape=(784,), num_classes=10):
    """
    Build and compile a deep neural network with Keras and TensorFlow.
    :param input_shape: input shape of each sample. Each image is 28x28 flattened to 784x1
    :param num_classes: number of different lables. We have 10 labels (0-9) for MNIST
    :return: the compiled model
    """
    # Create the deep neural network
    model = Sequential()
    model.add(Dense(60, activation='relu', input_shape=input_shape))
    model.add(Dense(60, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # Print summary and compile
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def test_my_img(img_name, show_img=False):
    """
    Test an image from your local drive. Add the image in the folder: my_imgs/.
    :param img_name: image name in the folder my_imgs
    :param show_img: if true, the picture sent to the model is displayed
    """
    my_image = get_my_image('my_imgs/' + img_name)
    if show_img:
        print("Showing local image file:", img_name)
        plot_image(my_image[0], title=img_name)
    my_image = np.array([my_image[0].flatten()])
    my_image /= 255.
    my_pred = model.predict_proba(my_image)
    print("My pred (" + img_name + "):", np.argmax(my_pred))
    print("My pred probability (" + img_name + "):", my_pred[0])


# Parameters
img_rows = 28
img_cols = 28
n_classes = 10
batch_size = 128
epochs = 5

# Fetch the dataset
x_train, x_test, y_train, y_test = get_mnist(show_example=False)

# Preprocess the dataset
x_train, x_test, y_train, y_test = preprocess_dataset(x_train, x_test, y_train, y_test, num_classes=n_classes)

# Build and compile the model
model = build_model(num_classes=n_classes)

# For visualizing training in TensorBoard
# Open terminal and cd into this folder (FagkveldML). Type: tensorboard --logdir .\Graph\
callback = callbacks.TensorBoard(log_dir='./Graph/' + strftime("%Y_%m_%d_%H_%M_%S", gmtime()), histogram_freq=0,
                                 write_graph=True, write_images=True)

# Train the model with training set
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.1, callbacks=[callback])

# Evaluate the model on test data
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Nicer prints to console
np.set_printoptions(precision=3, suppress=True)

# Test an image on your local drive
test_my_img('my_two.png', show_img=True)
