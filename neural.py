
import csv
import matplotlib.pyplot as plt
import time
import numpy as np
import sys

from random import shuffle

from sklearn import neural_network, metrics, model_selection
from datasets import digits, wave

# General Options
TITLE = 'Neural Network Classifier'
DATASET = digits

# Parameter Options
LEARNING_RATE_EXPONENT_MIN = 1
LEARNING_RATE_EXPONENT_MAX = 30
TOLERANCE_EXPONENT_MIN = 20
TOLERANCE_EXPONENT_MAX = 40


LAYERS_TOPOLOGY = (30,)

# Final Report Options
LEARNING_RATE = 1e-2
TOLERANCE = 1e-4



def parametre(dataset, blocking = True):
    plot_x = []
    plot_y_cv = []
    plot_y_training = []

    for learning_rate_exponent in range(LEARNING_RATE_EXPONENT_MIN, LEARNING_RATE_EXPONENT_MAX):
        learning_rate = 10**(-learning_rate_exponent / 10)
        print("Computing score for learning_rate = {} ...".format(learning_rate))
        classifier = neural_network.MLPClassifier(
            solver='sgd',
            hidden_layer_sizes=LAYERS_TOPOLOGY,
            learning_rate_init=learning_rate)

        result = model_selection.cross_validate(
                classifier, 
                dataset.training_features, 
                dataset.training_labels, 
                cv=10, return_train_score=True)
        test_score = result['test_score']
        train_score = result['train_score']
        fit_time = result['fit_time']
        score_time = result['score_time']

        plot_x.append(learning_rate)
        plot_y_cv.append(1.0 - np.array(test_score).mean())
        plot_y_training.append(1.0 - np.array(train_score).mean())
    
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.grid()
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title(TITLE)
    plt.semilogx(plot_x, plot_y_cv)
    plt.semilogx(plot_x, plot_y_training)
    plt.legend(['Cross validation', 'Training'])

    plot_x = []
    plot_y_cv = []
    plot_y_training = []

    for tolerance_exponent in range(TOLERANCE_EXPONENT_MIN, TOLERANCE_EXPONENT_MAX):
        tolerance = 10**(-tolerance_exponent / 10)
        print("Computing score for tolerance = {} ...".format(tolerance))
        classifier = neural_network.MLPClassifier(
            solver='sgd',
            hidden_layer_sizes=LAYERS_TOPOLOGY,
            tol=tolerance)
        
        result = model_selection.cross_validate(
                classifier, 
                dataset.training_features, 
                dataset.training_labels, 
                cv=10, return_train_score=True)
        test_score = result['test_score']
        train_score = result['train_score']
        fit_time = result['fit_time']
        score_time = result['score_time']

        plot_x.append(tolerance)
        plot_y_cv.append(1.0 - np.array(test_score).mean())
        plot_y_training.append(1.0 - np.array(train_score).mean())
    
    plt.subplot(1, 2, 2)
    plt.grid()
    plt.xlabel('Tolerance')
    plt.ylabel('Loss')
    plt.title(TITLE)
    plt.semilogx(plot_x, plot_y_cv)
    plt.semilogx(plot_x, plot_y_training)
    plt.legend(['Cross validation', 'Training'])

    plot_x = []
    plot_y_cv = []
    plot_y_training = []
    if blocking:
        plt.show()
    

def learning(dataset, blocking = True):
    classifier = neural_network.MLPClassifier()

    plot_x = []
    plot_y_testing = []
    plot_y_mean = []

    train_sizes = np.linspace(0.1, 0.9, 30)

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
    classifier = neural_network.MLPClassifier(activation='tanh',learning_rate_init=LEARNING_RATE, tol=TOLERANCE)

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
        


