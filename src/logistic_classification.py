'''
This script trains a logistic classification model and saves the model and classification report
'''

# import packages and modules
import os
import argparse
from joblib import dump, load
from sklearn.linear_model import LogisticRegression

# from my own utils script, import functions
from clf_utils import load_and_split_data, save_classification_report

# define argument parser
def argument_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--X_name', type=str, help= 'Name of vectorized X matrix. Must be a .npz file saved in the /in folder', default='X_vect.npz')
    parser.add_argument('--y_name', type=str, help= 'Name of array containing y labels. Must be a .npy file saved in the /in folder', default='y.npy')
    parser.add_argument('--report_name', type=str, help = 'Name of the output classification report', default='logistic_classification_report.txt')

    args = vars(parser.parse_args())
    
    return args

# fit model and predict on test data
def fit_and_predict(X_train: scipy.sparse._csr.csr_matrix, X_test: scipy.sparse._csr.csr_matrix, y_train: numpy.ndarray) -> numpy.ndarray:
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
    # create logistic classifier
    classifier = LogisticRegression(random_state=2830).fit(X_train, y_train)

    # save classifier in 'models' folder
    dump(classifier, os.path.join(models, "LR_classifier.joblib"))

    # predict on test data
    y_pred = classifier.predict(X_test)

    return y_pred

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




