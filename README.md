# Assignment 2 - Text classification benchmarks

This assignment is the second assignment for the portfolio exam in the Language Analytics course at Aarhus University, spring 2024.

### Contributions
All code was created by me, but code provided in the notebooks for the course has been reused. 

### Assignment description

For this exercise, you should write *two different scripts*. One script should train a logistic regression classifier on the data; the second script should train a neural network on the same dataset. Both scripts should do the following:

- Save the classification report as a text file to the folder called ```out```
- Save the trained models and vectorizers to the folder called ```models```

### Contents of the repository

| <div style="width:120px"></div>| Description |
|---------|:-----------|
|```data.zip```| Zipped file containing the *fake_or_real_news.csv* csv file used for the assignment as well as the Tfidf-vectorized text column (*X_vect.npz*) and the corresponding labels (*y.npy*). Unzipping it creates a folder called ```in```, which contains these files used for the analysis |
| ```out``` | Contains the output classification reports produced by running the two classification models as well as a plot of the loss curve for the neural network classification model |
| ```src```  | Contains the Python scripts for vectorizing the data, utils for classification as well as creating and fitting logistic and neural network classification models |
|```models```| Saved Tfidf-vectorizer, logistic and neural network classification models |
| ```run.sh```    | Bash script for running all code with default arguments |
| ```setup.sh```  | Bash script for setting up virtual environment |
| ```requirements.txt```  | Packages required to run the code|
|```emissions```|Contains csv files with information about how much CO2 is emitted when running the code, which is used for [Assignment 5](https://github.com/louisebphansen/assignment-5-evaluating-environmental-impact-louisebphansen)|

### Methods
This project contains the code to vectorize a dataset of fake and real news articles and use this vectorized text as the input to two seperate classifiers, predicting whether the text is fake or real news. 

More specifically, the script ```src/vectorize.py``` creates a Tf-Idf vectorizer using SciKit-Learn. This vectorizer is then used to vectorize the text in the dataset. The Tf-Idf vectorizer is saved in the ```models```folder and the vectorized text is saved in the ```in``` folder (which is created once the data.zip is unzipped).

```src/clf_utils.py``` contains util functions for classification. These include functions to load the vectorized dataset and split this into train and test sets (80/20 split) as well as a function for creating and saving a classification report.

```logistic_classification.py``` and ```nn_classification.py``` creates and fits a logistic regression classification model and a neural network classification model, respectively, on the vectorized and train-test split dataset. The model is trained on the train-split of the data whereas the test-split is used for predicting. The output classification reports of both scripts are saved in the ```out```folder, showcasing trues versus predicted labels for the test dataset. The fitted models are saved in the ```models```folder.

### Data
The data used for this assignment is the *fake_or_real_news* dataset, which consists of 6335 news stories labelled as either fake or real news. The dataset can be found in [this link](https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news). The data can be found in this repository by unzipping *data.zip*. See the **Usage** section for specifications.

### Usage

All code for this assignment was designed to run on an Ubuntu 24.04 operating system using Python version 3.12.2. It is therefore not guaranteed that it will work on other operating systems.

It is important that you run all code from the main folder, i.e., *assignment-2-text-classification-louisebphansen*. Your terminal should look like this:

```
--your_path-- % assignment-2-text-classification-louisebphansen %
```

#### Set up virtual environment

To run the code in this repo, clone it using ```git clone```.

In order to set up the virtual environment, the *venv* package for Python needs to be installed first:

```
sudo apt-get update

sudo apt-get install python3-venv
```

Next, run:

```
bash setup.sh
```

This will create a virtual environment in the directory (```env```) and install the required packages to run the code.


#### Run code

To run the code, you can do the following:

##### Run script with predefined arguments

To run the code in this repo with predefined/default arguments, run:
```
bash run.sh
```

This will activate the virual environment and unzip the *data.zip* file to create the ```in``` folder containing the *fake_or_real_news.csv* data. Next, the ```src/vectorize.py```, ```src/logistic_classification.py``` and ```src/nn_classification.py``` scripts will be run with default arguments. The vectorized text is saved in the newly created ```in``` folder, and the classification reports and loss curve plot are saved in the ```out```folder.

##### Define arguments yourself
Alternatively, the script(s) can be run seperately with different arguments:

```
# activate the virtual environment
source ./env/bin/activate

# unzip data folder to create /in folder containing the data
unzip data.zip

python3 src/vectorize.py --dataset <dataset> --lowercase <lowercase> --max_df <max_df> --min_df <min_df> --max_features <max_features>
```
**Arguments:**

- **Dataset:** Name of dataset placed in the ```in``` folder. Must be a csv file containing columns 'text' (input variable) and 'label' (target variable). Default: fake_or_real_news.csv

- **Lowercase:** Whether the text should be converted to lowercase when vectorized. Default: True

- **Max_df:** Maximum document frequency. Default: 0.95

- **Min_df:** Minimum document frequency. Default: 0.05

- **Max_features:** Maximum number of features to include. Default: 500


```
# activate the virtual environment
source ./env/bin/activate

# unzip data folder to create /in folder containing the data
unzip data.zip

python3 src/logistic_classification.py --X_name <X_name> --y_name <y_name> --report_name <report_name>
```

**Arguments:**

- **X_name:** Name of the saved and vectorized text column. Must be a .npz file saved in the /in folder. Default: X_vect.npz

- **Y_name:** Name of array containing labels. Must be a .npy file saved in the /in folder'. Default: y.npy

- **Report_name:** Name of the output classification report. Default: logistic_classification_report.txt


```
# activate the virtual environment
source env/bin/activate

# unzip data folder to create /in folder containing the data
unzip data.zip

python3 src/nn_classification.py --X_name <X_name> --y_name <y_name> --activation_function <activation_function> --hidden_layer_sizes  <hidden_layer_sizes> --report_name <report_name>
```
**Arguments:**

- **X_name:** Name of the saved and vectorized text column. Must be a .npz file saved in the /in folder. Default: X_vect.npz

- **Y_name:** Name of array containing labels. Must be a .npy file saved in the /in folder'. Default: y.npy

- **Activation_function:** What activation function to use for the classifier. Default: logistic

- **Hidden_layer_sizes:** Specify the size(s) of the hidden layer(s). No default (this parameters needs to be defined)

- **Report_name:** Name of the output classification report. Default: neuralnet_classification_report.txt


### Results
The tables below showcase the classification report for the logistic classifion and the report for the neural network classification of the *fake_or_real_news* dataset saved in the ```out``` folder.

**Logistic Classification** 

![Screenshot 2024-03-04 at 15 12 09](https://github.com/louisebphansen/assignment-2-text-classification-louisebphansen/assets/75262659/dd6918b5-3097-4970-b55a-dccc627b04c3)



**Neural Network Classification**

![Screenshot 2024-03-04 at 15 12 19](https://github.com/louisebphansen/assignment-2-text-classification-louisebphansen/assets/75262659/2f14f287-a52e-4db1-9f48-fee4643337b2)

**Loss Curve for Neural Network Classifier**


### Discussion
The results from the two classification models show that there is hardly any difference between running the logistic regression and neural network classifier for this dataset. 
Interestingly, this indicates that the added complexity in the neural network classifier compared to the logistic classification does not result in better performance accuracies. The results thus demonstrates that more complex, computationally expensive models are not always the answer, especially not for more simple, binary problems such as the problem on hand. This further becomes an important notion when relating these results to their carbon emissions as done in [Assignment 5](https://github.com/louisebphansen/assignment-5-evaluating-environmental-impact-louisebphansen.git).

#### Limitations
When considering the limitations of this approach, it is important to mention that TF-IDF vectorizing is not necessarily the best way to represent text to a model. One could consider using more sophisticated approaches, for example by using word embedding models such as word2vec or GloVe or transformers such as BERT. This could yield more accurate and contextual representations of the input texts. However, as the above results show that we actually get a decent result using TF-IDF vectorization, added complexity may yet again not be the answer.



### A note on carbon emissions

