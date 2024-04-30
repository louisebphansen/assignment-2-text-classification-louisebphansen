'''
LANGUAGE ANALYTICS @ AARHUS UNIVERSITY, ASSIGNMENT 2: Text classification

AUTHOR: Louise Brix Pilegaard Hansen

DESCRIPTION:
This script takes a csv file as the input and vectorizes a column of text. The vectorized text and beloning labels are saved in the '/in' folder.
'''
# import packages and modules
import os
import argparse
from joblib import dump, load
import pandas as pd
import numpy as np
import scipy as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from codecarbon import EmissionsTracker
from codecarbon import track_emissions

# define emissionstracker to track CO2 emissions (for assignment 5)
tracker = EmissionsTracker(project_name="assignment2_vectorize_subtasks",
                           experiment_id="vectorizing",
                           output_dir='emissions',
                           output_file="emissions_vectorize.csv")

# define argument parser
def argument_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, help= 'name of the linguistic dataset to use. must be a csv file in the /in folder', default='fake_or_real_news.csv')
    parser.add_argument('--lowercase', type=bool,help ='whether to lowercase the text or not', default=True)
    parser.add_argument('--max_df', type=float, help='set max document frequency', default = 0.95)
    parser.add_argument('--min_df', type=float, help='set min document frequency', default = 0.05)
    parser.add_argument('--max_features', type=int, help='how many features to include', default = 500)

    args = vars(parser.parse_args())
    
    return args

# load data
def data_loader(csv_name: str) -> pd.DataFrame:

    '''
    Load csv file from input folder

    Arguments:
        - csv_name: Name of .csv file placed in /in folder containing text dataset
    
    Returns:
        - data: pandas dataframe

    '''

    # load dataframe from folder
    filename = os.path.join('in', csv_name)

    data = pd.read_csv(filename, index_col=0)

    return data

# vectorize data
def prep_data(data: pd.DataFrame, vectorizer: TfidfVectorizer):
    '''
    Vectorizes the text column (i.e., our X variable) of a pd.DataFrame and saves this as a .npz file in the /in folder.
    Also saves our y variable (i.e., labels) as a .npy file in the /in folder.
    The fitted vectorizer is saved in the /models folder.

    Arguments:
        - data: a pandas dataframe containing columns 'text' and 'label'
        - vectorizer: a sklearn TfidfVectorizer object
    
    Returns:
        None

    '''

    # save X and y variables from dataframe
    X = data['text']
    y = data['label']

    # use vectorizer to fit and transform X
    X_vect = vectorizer.fit_transform(X)

    # save vectorizer
    dump(vectorizer, os.path.join('models', "tfidf_vectorizer.joblib"))

    # save vectorized X variable
    sp.sparse.save_npz(os.path.join('in', 'X_vect.npz'), X_vect)

    # save y variable
    np.save(os.path.join('in', 'y.npy'), y)

def vectorize_data(dataset, lowercase, max_df, min_df, max_features):

    '''
    This function loads a csv file to a pandas dataframe, Tfidf-vectorizes it and saved the vectorized
    data in the /in folder.

    Arguments:
        - dataset: name of csv file placed in the /in folder containing dataset to be vectorized
        - lowercase: whether to lowercase the text (bool)
        - max_df: max document frequency  
        - min_df: minimum document frequency 
        - max_features: how many features to include from corpus
    
    Returns: 
        None
    '''

    # load data
    data = data_loader(dataset)

    # create vectorizer from custom arguments
    vectorizer = TfidfVectorizer(ngram_range = (1,2), # allow 2-word sequences
                             lowercase = lowercase, # whether to lowercase the text       
                             max_df = max_df, # max document frequency          
                             min_df = min_df, # minimum document frequency          
                             max_features = max_features) # how many features to include from corpus

    # track vectorize task
    tracker.start_task('Vectorizing data')

    # vectorize and save data
    prep_data(data, vectorizer)

    # stop task tracker and overall tracker
    vectorize_emissions = tracker.stop_task()
    tracker.stop()

# create new tracker using a decorator to track emissions for running the entire script
@track_emissions(project_name="assignment2_vectorize_full",
                experiment_id="vectorize_full",
                output_dir='emissions',
                output_file="emissions_vectorize_FULL.csv")
def main():
    # parse arguments
    args = argument_parser()

    # load dataset, vectorize it and save TfIDF-vectorized dataset
    vectorize_data(dataset=args['dataset'], 
                    lowercase=args['lowercase'], 
                    max_df=args['max_df'], 
                    min_df=args['min_df'], 
                    max_features=args['max_features'])

if __name__ == '__main__':
   main()


