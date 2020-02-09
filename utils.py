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
