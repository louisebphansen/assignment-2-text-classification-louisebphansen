'''
This script takes a dataframe (csv file) as the input and vectorizes a column of text. The vectorized text
and beloning labels are saved in the '/in' folder.
'''
# import packages and modules
import os
import argparse
from joblib import dump, load
import pandas as pd
import numpy as np
import scipy as sp
from sklearn.feature_extraction.text import TfidfVectorizer

# define argument parser
def argument_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, help= 'name of the linguistic dataset to use. must be a csv file in the /in folder', default='fake_or_real_news.csv')
    parser.add_argument('--lowercase', type=bool,help ='whether to lowercase the text or not', default=True)
    parser.add_argument('--max_df', type=float, help='set max document frequency', default = 0.95)
    parser.add_argument('--min_df', type=float, help='set min document frequency', default = 0.05)
    parser.add_argument('--max_features', type=int, help='how many features to include' ,default = 500)

    args = vars(parser.parse_args())
    
    return args

# load data
def data_loader(csv_name: str) -> pd.DataFrame:

    '''
    Load csv file from input folder

    Arguments:
        - csv_name: Name of .csv file containing text dataset
    
    Returns:
        - data: pandas dataframe

    '''

    # load dataframe from folder
    filename = os.path.join('in', csv_name)

    data = pd.read_csv(filename, index_col=0)

    return data

# vectorize data
def prep_data(data: pd.DataFrame, vectorizer: sklearn.feature_extraction.text.TfidfVectorizer):
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

def main():
    
    # parse arguments
    args = argument_parser()

    # load data
    data = data_loader(args['dataset'])

    # create vectorizer from custom arguments
    vectorizer = TfidfVectorizer(ngram_range = (1,2),
                             lowercase = args['lowercase'],       
                             max_df = args['max_df'],           
                             min_df = args['min_df'],           
                             max_features = args['max_features']) 

    # vectorize and save data
    prep_data(data, vectorizer)

if __name__ == '__main__':
   main()


