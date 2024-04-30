'''
LANGUAGE ANALYTICS @ AARHUS UNIVERSITY, ASSIGNMENT 2: Text classification

AUTHOR: Louise Brix Pilegaard Hansen

DESCRIPTION:
This script trains a logistic classification model and saves the model and classification report
'''

# import packages and modules
import os
import argparse
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
import scipy as sp
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from codecarbon import EmissionsTracker
from codecarbon import track_emissions

# from my own utils script for this repo, import functions
from clf_utils import load_and_split_data, save_classification_report

# define emissionstracker to track CO2 emissions (for assignment 5)
tracker = EmissionsTracker(project_name="assignment2_logistic_regression_subtasks",
                           experiment_id="logistic_regression",
                           output_dir='emissions',
                           output_file="emissions_logistic_regression.csv")

# define argument parser
def argument_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--X_name', type=str, help= 'Name of vectorized X matrix. Must be a .npz file saved in the /in folder', default='X_vect.npz')
    parser.add_argument('--y_name', type=str, help= 'Name of array containing y labels. Must be a .npy file saved in the /in folder', default='y.npy')
    parser.add_argument('--report_name', type=str, help = 'Name of the output classification report', default='logistic_classification_report.txt')

    args = vars(parser.parse_args())
    
    return args

# fit model and predict on test data
def fit_and_predict(X_train, X_test, y_train: np.ndarray) -> np.ndarray:
    '''
    Creates a logistic regression classifier and fits it to training data.
    The classifier is saved and used to predict on test data.

    Arguments:
        - X_train: Training data used for fitting the model
        - y_train: Target training data used for fitting the model
        - X_test: Data used for predictions
    
    Returns:
        - y_pred: Predicted y-labels by the model for the X_test data
    
    '''

    # track emissions for model fitting task
    tracker.start_task('Fitting logistic model')

    # create logistic classifier
    classifier = LogisticRegression(random_state=2830).fit(X_train, y_train)

    # stop emission of task when fitting is done
    fitting_emissions = tracker.stop_task()

    # save classifier in 'models' folder
    dump(classifier, os.path.join('models', "LR_classifier.joblib"))

    # track emissions for model prediction
    tracker.start_task('Predict using logistic model')

    # predict on test data
    y_pred = classifier.predict(X_test)

    # stop tracker for task
    predict_emissions = tracker.stop_task()

    # stop tracker completely
    tracker.stop()

    return y_pred

# create new tracker using a decorator to track emissions for running the entire script
@track_emissions(project_name="assignment2_logistic_full",
                experiment_id="logistic_regression_full",
                output_dir='emissions',
                output_file="emissions_logistic_regression_FULL.csv")

def main():
    # parse arguments
   args = argument_parser()

    # load data and split into train and test
   X_train, X_test, y_train, y_test = load_and_split_data(args['X_name'], args['y_name'])

    # fit model on training data and predict from test data
   y_pred = fit_and_predict(X_train, X_test, y_train)

    # save classification report
   save_classification_report(y_test, y_pred, args['report_name'])

if __name__ == '__main__':
   main()

