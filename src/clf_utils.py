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

def plot_cv_results(estimator, title, X, y, image_name, axes=None, ylim=None, cv=None,
                        n_jobs=None, scoring="accuracy", train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Function to run cross-validation on an estimator and plot the results.
    Saves the resulting plot in the /out folder.

    NB: The code for this function is taken from the 'classifier_utils.py' script provided for the Language Analytics course.
    See https://github.com/CDS-AU-DK/cds-language/blob/main/utils/classifier_utils.py

    Arguments: 
        - estimator: scikit-learn estimator, e.g. Logistic Regression estimator object
        - title: name of the plot
        - X: vectorized training data
        - y: Target data
        - cv: Cross-validation method
    
    Returns:
        None

    """
    if axes is None:
        _, axes = plt.subplots(figsize=(7, 7))

    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       scoring=scoring,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes.legend(loc="best")

    out_path = os.path.join("out", image_name)

    plt.savefig(out_path)

def cross_validate(X_name, y_name, estimator, classification_method, n_splits, image_name):
    '''
    Loads the full X and y data and performs cross validation 
    
    '''

    X_vect = sp.sparse.load_npz(os.path.join('in', X_name))
    y = np.load(os.path.join('in', y_name), allow_pickle=True)

    # run cross validation and save plots
    title = f"Learning Curves ({classification_method})"
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=2830)

    plot_cv_results(estimator, title, X_vect, y, image_name, cv=cv, n_jobs=4)