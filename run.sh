source env/bin/activate

unzip data.zip

python3 src/vectorize.py

python3 src/logistic_classification.py

python3 src/nn_classification.py --hidden_layer_sizes 50

deactivate