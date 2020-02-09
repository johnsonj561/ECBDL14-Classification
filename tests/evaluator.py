import os
import sys
import json
import pandas as pd
import tensorflow as tf
Keras = tf.keras
model_from_json = Keras.models.model_from_json
from sklearn.metrics import confusion_matrix, f1_score, precision_score, roc_auc_score
from sklearn.exceptions import UndefinedMetricWarning
import warnings
import numpy as np
import math
ecbdl14_root = '/home/jjohn273/git/ECBDL14-Classification/'
sys.path.append(ecbdl14_root)
from utils import load_model, rounded_str, get_best_threshold, write_performance_metrics


# input config
data_file = '/home/jjohn273/git/ECBDL14-Classification/data/ecbdl14.onehot.sample.hdf'
test_key = 'test'
train_key = 'train'
models_dir = '/home/jjohn273/git/ECBDL14-Classification/tests/trained-models/'

# output config
results_file = 'results.csv'


test_data = pd.read_hdf(data_file, test_key)
test_y, test_x = test_data['target'], test_data.drop(columns=['target'])

# training data will be used to estimate optimal thresholds
train_data = pd.read_hdf(data_file, train_key)
train_y, train_x = train_data['target'], train_data.drop(columns=['target'])


groups = [dir_name for dir_name in os.listdir(models_dir) if 'group' in dir_name]
group_paths = [os.path.join(models_dir, g) for g in groups]


# load model architecture
model_path = os.path.join(group_paths[0], 'model-architecture.json')
with open(model_path, 'r') as json_in:
    model_json = json_in.read()


# build results set
counts = {}
trained_models = []
for group_path in group_paths:
    weight_files = [path for path in os.listdir(os.path.join(group_path, 'weights')) if 'model.h5' in path]
    results = ([(*f.split('-')[:2], os.path.join(group_path, 'weights', f)) for f in weight_files])
    trained_models.extend(results)
    for (pos_size, run, weights_path) in results:
        counts[pos_size] = counts.get(pos_size, 0) + 1

trained_models = pd.DataFrame(trained_models, columns=['pos_size', 'run', 'path']) \
    .astype({'run': 'int32' }) \
    .sort_values(by=['pos_size', 'run'])


# skip completed rows
skip_rows = ['0.5%', '1%', '2%', '10%']

# make predictions with each trained model
for (pos_size, run, weights_file) in trained_models.values:
    if pos_size in skip_rows:
        continue
    if counts[pos_size] < 30:
        continue
    if run == 0:
        print(f'Starting {pos_size}')
    if run % 10 == 0:
        print(f'Starting run {run}')

    model = load_model(model_json, weights_file)

    # optimal threshold is estimated using the training data
    minority_size = float(pos_size.replace('%', ''))
    delta = 0.005 if minority_size > 1 else 0.001
    optimal_threshold = round(get_best_threshold(train_x, train_y, model, delta), 4)

    # theoretical threshold is the positive class prior
    theoretical_threshold = minority_size / 100

    default_threshold = 0.5

    # record results
    y_prob = model.predict(test_x)
    write_performance_metrics(test_y, y_prob, minority_size, 'optimal', optimal_threshold, results_file)
    write_performance_metrics(test_y, y_prob, minority_size, 'theoretical', theoretical_threshold, results_file)
    write_performance_metrics(test_y, y_prob, minority_size, 'default', default_threshold, results_file)
