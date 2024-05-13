# activate virtual environment
source env/bin/activate

# unzip data file (create /in folder with data)
unzip data.zip

# vectorize data
python3 src/vectorize.py

# run logistic regression classifier
python3 src/logistic_classification.py

# run neural network classification
python3 src/nn_classification.py --hidden_layer_sizes 50

# deactivate venv
deactivate