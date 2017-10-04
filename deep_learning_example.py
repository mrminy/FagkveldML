from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.utils import to_categorical

from data_utils import get_mnist

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
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)

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
    # Create network
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=input_shape))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # Print summary and compile
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


n_classes = 10
batch_size = 30
epochs = 2

x_train, x_test, y_train, y_test = get_mnist()
x_train, x_test, y_train, y_test = preprocess_dataset(x_train, x_test, y_train, y_test, num_classes=n_classes)
model = build_model(num_classes=n_classes)
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.1)

preds = model.predict(x_test)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
