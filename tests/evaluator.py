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

# caused by divide by zero during metrics calcultiong
# ignoring  because it is saturating the error logs
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)



def load_model(model_json, weights_path):
    model = model_from_json(model_json)
    model.load_weights(weights_path)
    return model


def rounded_str(num, precision=6):
    if type(num) is str:
        return num
    return str(round(num, precision))


def get_best_threshold(train_x, train_y, model, delta=0.005):
    curr_thresh = 0.0
    best_thresh = 0.0
    best_gmean = 0.0

    y_prob = model.predict(train_x)

    while True:
        y_pred = np.where(y_prob > curr_thresh, np.ones_like(y_prob), np.zeros_like(y_prob))
        tn, fp, fn, tp = confusion_matrix(train_y, y_pred).ravel()
        tpr = (tp) / (tp + fn)
        tnr = (tn) / (tn + fp)
        if tnr > tpr:
            return best_thresh
        gmean = math.sqrt(tpr*tnr)
        if gmean > best_gmean:
            best_gmean = gmean
            best_thresh = curr_thresh
        curr_thresh += delta

    return best_thresh



columns = ['minority_size', 'strategy', 'threshold', 'tp', 'fp', 'tn', 'fn', 'tpr', 'tnr', 'roc_auc', 'geometric_mean', 'arithmetic_mean', 'f1_score', 'precision']


def write_performance_metrics(y_true, y_prob, minority_size, strategy, threshold, path):
    # include header if file doesn't exist yet
    out = ",".join(columns) if not os.path.isfile(path) else ''

    predictions = np.where(y_prob > threshold, 1.0, 0.0)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
    tpr = (tp) / (tp + fn)
    tnr = (tn) / (tn + fp)
    roc_auc = roc_auc_score(y_true, y_prob)
    geometric_mean = math.sqrt(tpr * tnr)
    arithmetic_mean = 0.5 * (tpr + tnr)
    f1 = f1_score(y_true, predictions)
    precision = precision_score(y_true, predictions)

    results = [minority_size, strategy, threshold,
               tp, fp, tn, fn, tpr, tnr, roc_auc,
               geometric_mean, arithmetic_mean, f1, precision]

    results = [rounded_str(x) for x in results]

    out += '\n' + ','.join(results)

    with open(path, 'a') as outfile:
        outfile.write(out)


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


# make predictions with each trained model
for (pos_size, run, weights_file) in trained_models.values:
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
