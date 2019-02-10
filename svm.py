
import csv
import matplotlib.pyplot as plt
import time
import numpy as np
import sys

from random import shuffle

from sklearn import svm, metrics, model_selection
from datasets import digits, wave

# General Options
TITLE = 'SVM Classifier'
DATASET = digits
KERNEL_TYPE = 'linear'
KERNEL_DEGREE = 3


# Final Report Options
FINAL_KERNEL_C = 1
FINAL_KERNEL_TYPE = 'linear'
FINAL_KERNEL_DEGREE = 3




def learning(dataset, blocking = True):
    classifier = svm.SVC(C=FINAL_KERNEL_C, kernel=FINAL_KERNEL_TYPE, degree=FINAL_KERNEL_DEGREE)

    plot_x = []
    plot_y_testing = []
    plot_y_mean = []

    train_sizes = np.linspace(0.1, 0.9, 10)

    train_size_abs, train_scores, test_scores = model_selection.learning_curve(classifier, dataset.training_features, dataset.training_labels,
        cv=10, train_sizes=train_sizes)
    
    train_losses = [1 - np.array(a).mean() for a in train_scores]
    test_losses = [1 - np.array(a).mean() for a in test_scores]
    
    plt.figure()
    plt.grid()
    plt.xlabel('Training Set Size')
    plt.ylabel('Loss')
    plt.title(TITLE)
    plt.plot(train_size_abs, train_losses)
    plt.plot(train_size_abs, test_losses)
    plt.legend(['Training', 'Testing'])
    if blocking:
        plt.show()


def final_score(dataset):
    classifier = svm.SVC(C=FINAL_KERNEL_C, kernel=FINAL_KERNEL_TYPE, degree=FINAL_KERNEL_DEGREE)

    classifier.fit(dataset.training_features, dataset.training_labels)

    predicted = classifier.predict(dataset.testing_features)

    print(metrics.classification_report(dataset.testing_labels, predicted))
    print(metrics.confusion_matrix(dataset.testing_labels, predicted))

if __name__ == '__main__':
        mode = sys.argv[1]
        if mode == 'final':
            final_score(DATASET)
      
        elif mode == 'learning':
            learning(DATASET)
       



