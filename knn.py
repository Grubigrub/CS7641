
import csv
import matplotlib.pyplot as plt
import time
import numpy as np
import sys

from random import shuffle

from sklearn import neighbors, metrics, model_selection
from datasets import digits , wave

# General Options
TITLE = 'kNN Classifier'
DATASET = wave

#Parametre Options
K_MIN = 1
K_MAX = 20

# Final Report Options
NEIGHBORS_COUNT = 3

def learning(dataset, blocking = True):
    classifier = neighbors.KNeighborsClassifier()

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

def parametre(dataset, blocking = True):
    plot_x = []
    plot_y_cv = []
    plot_y_training = []

    for k in range(K_MIN, K_MAX):
        print("Computing score for k = {} ...".format(k))
        classifier = neighbors.KNeighborsClassifier(n_neighbors=k)

        result = model_selection.cross_validate(
                classifier, 
                dataset.training_features, 
                dataset.training_labels, 
                cv=10, return_train_score=True)
        test_score = result['test_score']
        train_score = result['train_score']
        fit_time = result['fit_time']
        score_time = result['score_time']

        plot_x.append(k)
        plot_y_cv.append(1.0 - np.array(test_score).mean())
        plot_y_training.append(1.0 - np.array(train_score).mean())
    
    plt.figure()
    plt.grid()
    plt.xlabel('Neighbors Count')
    plt.ylabel('Loss')
    plt.title(TITLE)
    plt.plot(plot_x, plot_y_cv)
    plt.plot(plot_x, plot_y_training)
    plt.legend(['Cross validation', 'Training'])
    if blocking:
        plt.show()



def final_score(dataset):
    classifier = neighbors.KNeighborsClassifier(n_neighbors=NEIGHBORS_COUNT)

    classifier.fit(dataset.training_features, dataset.training_labels)

    predicted = classifier.predict(dataset.testing_features)

    print(metrics.classification_report(dataset.testing_labels, predicted))
    print(metrics.confusion_matrix(dataset.testing_labels, predicted))

if __name__ == '__main__':
        mode = sys.argv[1]
        if mode == 'final':
            final_score(DATASET)
        elif mode == 'parametre':
            parametre(DATASET)
        elif mode == 'learning':
            learning(DATASET)
        


