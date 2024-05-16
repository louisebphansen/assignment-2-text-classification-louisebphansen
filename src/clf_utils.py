'''
LANGUAGE ANALYTICS @ AARHUS UNIVERSITY, ASSIGNMENT 2: Text classification

AUTHOR: Louise Brix Pilegaard Hansen

DESCRIPTION:
This script contains classification utils for splitting into train and test data as well as saving the classification report.
'''
# import packages and modules
import os
import scipy as sp
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics

# load and split data from folder
def load_and_split_data(X_name:str, y_name:str) -> np.ndarray:
    '''
    Loads vectorized X and y variables from /in folder and splits them into training and testing data.

    Arguments:
        - X_name: name of .npz file saved in /in folder
        - y_name: name of .npy file saved in /in folder
    
    Returns:
        Scipy sparse matrices for X_train, X_test and np.arrays for y_train, y_test
    '''
    # load X and y variables from file
    X = sp.sparse.load_npz(os.path.join('in', X_name))
    y = np.load(os.path.join('in', y_name), allow_pickle=True)

    # split into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X,           
                                                        y,        
                                                        test_size=0.2,   
                                                        random_state=2830)

    return X_train, X_test, y_train, y_test

# save classification report
def save_classification_report(y_test: np.ndarray, y_pred: np.ndarray, report_name: str):
    '''
    Create and save classification report.

    Arguments:
        - y_test: The actual y_labels
        - y_pred: Predicted y_labels
        - report_name: What to call the classification report

    Returns:
        None 
    
    '''
    # create classification report
    classification_metrics = metrics.classification_report(y_test, y_pred)
    
    # save report
    out_path = os.path.join("out", report_name)

    with open(out_path, 'w') as file:
                file.write(classification_metrics)