
import csv
import matplotlib.pyplot as plt
import time
import numpy as np
import sys

from random import shuffle

from sklearn import tree, metrics, model_selection
from datasets import digits, wave

# General Options
TITLE = 'Decision Tree Classifier'
DATASET = digits

# Parameter Options
DEPTH_MIN = 2
DEPTH_MAX = 30
LEAF_NODES_MIN = 50
LEAF_NODES_MAX = 200


# Final Options
TREE_MAX_DEPTH = 10
TREE_MAX_LEAF_NODES = 140




def parametre(dataset, blocking = True):
    plot_x = []
    plot_y_cv = []
    plot_y_training = []

    for max_depth in range(DEPTH_MIN, DEPTH_MAX):
        print("Computing score for max_depth = {} ...".format(max_depth))
        classifier = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)

        result = model_selection.cross_validate(
                classifier, 
                dataset.training_features, 
                dataset.training_labels, 
                cv=10, return_train_score=True)
        test_score = result['test_score']
        train_score = result['train_score']
        fit_time = result['fit_time']
        score_time = result['score_time']

        plot_x.append(max_depth)
        plot_y_cv.append(1.0 - np.array(test_score).mean())
        plot_y_training.append(1.0 - np.array(train_score).mean())
    
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.grid()
    plt.xlabel('Tree Max Depth')
    plt.ylabel('Loss')
    plt.title(TITLE)
    plt.plot(plot_x, plot_y_cv)
    plt.plot(plot_x, plot_y_training)
    plt.legend(['Cross validation', 'Training'])

    plot_x = []
    plot_y_cv = []
    plot_y_training = []

    for max_leaf_nodes in range(LEAF_NODES_MIN, LEAF_NODES_MAX):
        print("Computing score for max_leaf_nodes = {} ...".format(max_leaf_nodes))
        classifier = tree.DecisionTreeClassifier(criterion='entropy', max_leaf_nodes=max_leaf_nodes)

        result = model_selection.cross_validate(
                classifier, 
                dataset.training_features, 
                dataset.training_labels, 
                cv=10, return_train_score=True)
        test_score = result['test_score']
        train_score = result['train_score']
        fit_time = result['fit_time']
        score_time = result['score_time']

        plot_x.append(max_leaf_nodes)
        plot_y_cv.append(1.0 - np.array(test_score).mean())
        plot_y_training.append(1.0 - np.array(train_score).mean())
    
    plt.subplot(1, 2, 2)
    plt.grid()
    plt.xlabel('Tree Max Leaf Nodes')
    plt.ylabel('Loss')
    plt.title(TITLE)
    plt.plot(plot_x, plot_y_cv)
    plt.plot(plot_x, plot_y_training)
    plt.legend(['Cross validation', 'Training'])

    plot_x = []
    plot_y_cv = []
    plot_y_training = []

  
    if blocking:
        plt.show()
        
        
        
        
        
        
        

def learning(dataset, blocking = True):
    classifier = tree.DecisionTreeClassifier(criterion='entropy')

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
    classifier = tree.DecisionTreeClassifier(criterion='entropy', max_depth=TREE_MAX_DEPTH, max_leaf_nodes=TREE_MAX_LEAF_NODES)

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
    
        

 

