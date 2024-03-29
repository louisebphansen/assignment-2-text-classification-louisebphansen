'''
This script trains a neural network classification model and saves the model and classification report
'''

# import packages and modules
import os
import argparse
from joblib import dump, load
from sklearn.neural_network import MLPClassifier
import scipy as sp
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics

# from my own utils script, import functions
from clf_utils import load_and_split_data, save_classification_report

# define argument parser
def argument_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--X_name', type=str, help= 'Name of vectorized X matrix. Must be a .npz file saved in the /in folder', default='X_vect.npz')
    parser.add_argument('--y_name', type=str, help= 'Name of array containing y labels. Must be a .npy file saved in the /in folder', default='y.npy')
    parser.add_argument('--activation_function', type=str, help = 'what activation function to use for the classifier', default ='logistic')
    parser.add_argument("--hidden_layer_sizes", type=int, nargs='+', help='Specify size of hidden layers', default = 50)
    parser.add_argument('--report_name', type=str, help = 'Name of the output classification report', default='neuralnet_classification_report.txt')

    args = vars(parser.parse_args())
    
    return args

# fit model and predict on test data
def fit_and_predict(activation:str, hidden_layer_sizes:int, X_train, X_test, y_train: np.ndarray) -> np.ndarray:
    '''
    Creates a neural network classifier and fits it to training data.
    The classifier is saved and used to predict on test data.

    Arguments:
        - activation: what activation function to use
        - hidden_layer_sizes: sizes of hidden layer(s)
        - X_train: Training data used for fitting the model
        - y_train: Target training data used for fitting the model
        - X_test: Data used for predictions
    
    Returns:
        - y_pred: Predicted y-labels by the model for the X_test data
    '''

    # create neural network classifier
    classifier = MLPClassifier(activation = activation,
                           hidden_layer_sizes = tuple(hidden_layer_sizes),
                           max_iter=1000,
                           random_state = 2830)

    # fit on training data
    classifier.fit(X_train, y_train)

    # save classification model
    dump(classifier, os.path.join('models', "NN_classifier.joblib"))

    # predict on test data
    y_pred = classifier.predict(X_test)

    return y_pred

def main():

    # parse arguments
   args = argument_parser()

    # load data and split into test and train
   X_train, X_test, y_train, y_test = load_and_split_data(args['X_name'], args['y_name'])

    # fit classification model and predict on new data
   y_pred = fit_and_predict(args['activation_function'], args['hidden_layer_sizes'], X_train, X_test, y_train)

    # create and save classification report
   save_classification_report(y_test, y_pred, args['report_name'])

if __name__ == '__main__':
   main()