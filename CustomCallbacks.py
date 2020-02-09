from sklearn.metrics import roc_auc_score
import numpy as np
import tensorflow as tf
import sys
Callback = tf.keras.callbacks.Callback
ecbdl14_root = '/home/jjohn273/git/ECBDL14-Classification/'
sys.path.append(ecbdl14_root)
from utils import get_best_threshold


class KerasRocAucCallback(Callback):
    def __init__(self, x, y, is_validation=False, logger=None):
        super(Callback, self).__init__()
        self.x = x
        self.y = y
        self.auc_scores = []
        self.logger = logger
        self.metric_key = 'val_auc' if is_validation else 'train_auc'

    def on_epoch_end(self, epoch, logs={}, generator=None):
        probs = self.model.predict(self.x)
        auc = roc_auc_score(self.y, probs)
        self.auc_scores.append(auc)
        logs[self.metric_key] = auc
        if self.logger != None:
            self.logger.log_time(f'Epoch: {epoch}  {self.metric_key}: {auc}').write_to_file()

class KerasThresholdMonitoringCallback(Callback):
    def __init__(self, x, y, logger=None):
        super(Callback, self).__init__()
        self.x = x
        self.y = y
        self.logger = logger

    def on_epoch_end(self, epoch, logs={}, generator=None):
        optimal_threshold = get_best_threshold(self.x, self.y, self.model)
        logs['optimal_threshold'] = optimal_threshold
        if self.logger != None:
            self.logger.log_time(f'Completed run with threshold: {optimal_threshold}').write_to_file()
