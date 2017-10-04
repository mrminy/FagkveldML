from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from data_utils import get_simple_classification_dataset, get_classification_dataset
from matplotlib import pyplot as plt
from data_utils import plot_confusion_matrix

"""
Script for training and testing classical machine learning models on classification problems
"""


def train_and_score_model(clf, x_train, x_test, y_train, y_test):
    print(clf)
    clf.fit(x_train, y_train)

    preds = clf.predict(x_test)
    print("Labels:\t\t\t", y_test[:5])
    print("Predictions:\t", preds[:5])

    score = clf.score(x_test, y_test)
    print("Accuracy: %1.4f" % score)
    plot_confusion_matrix(y_test, preds)
    print("--------------")


# Fetch simple dataset
x_train, x_test, y_train, y_test = get_simple_classification_dataset(test_size=0.5)

# Uncomment this line to use another dataset
# x_train, x_test, y_train, y_test = get_classification_dataset(test_size=0.5)

# Do classification with different algorithms from sklearn
clf = KNeighborsClassifier()
train_and_score_model(clf, x_train, x_test, y_train, y_test)

clf = GaussianNB()
train_and_score_model(clf, x_train, x_test, y_train, y_test)

clf = DecisionTreeClassifier()
train_and_score_model(clf, x_train, x_test, y_train, y_test)

clf = RandomForestClassifier()
train_and_score_model(clf, x_train, x_test, y_train, y_test)