from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from data_utils import get_simple_classification_dataset, get_classification_dataset
from data_utils import plot_confusion_matrix

"""
Script for training and testing classical machine learning models on classification problems
"""


def train_and_score_model(clf, x_train, x_test, y_train, y_test, classes=None):
    """
    Trains a model (clf) with training set and evaluates on test set
    :param clf: an untrained classification model
    :param x_train: training set of input features
    :param x_test: test set of input features
    :param y_train: training set of labels
    :param y_test: test set of labels
    """
    classifier = type(clf).__name__
    print(classifier)
    clf.fit(x_train, y_train)

    preds = clf.predict(x_test)
    print("Labels:\t\t\t", y_test[:5])
    print("Predictions:\t", preds[:5])

    score = clf.score(x_test, y_test)
    print("Accuracy: %1.4f" % score)
    plot_confusion_matrix(y_test, preds, title=classifier, classes=classes)
    print("--------------")


# Fetch simple dataset (iris flower data)
x_train, x_test, y_train, y_test = get_simple_classification_dataset(test_size=0.5)
classes = ['setosa', 'versicolor', 'virginica']

# Uncomment next lines to use another dataset (breast cancer data)
# x_train, x_test, y_train, y_test = get_classification_dataset(test_size=0.5)
# classes = ['Cancerous', 'Noncancerous']

# Do classification with different algorithms from sklearn
clf = KNeighborsClassifier()
train_and_score_model(clf, x_train, x_test, y_train, y_test, classes=classes)

clf = GaussianNB()
train_and_score_model(clf, x_train, x_test, y_train, y_test, classes=classes)

clf = DecisionTreeClassifier()
train_and_score_model(clf, x_train, x_test, y_train, y_test, classes=classes)

clf = RandomForestClassifier()
train_and_score_model(clf, x_train, x_test, y_train, y_test, classes=classes)
