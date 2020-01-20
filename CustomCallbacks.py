from sklearn.metrics import roc_auc_score
import numpy as np
import tensorflow as tf
Callback = tf.keras.callbacks.Callback

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
