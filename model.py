from sklearn.metrics import roc_auc_score
import numpy as np
import tensorflow as tf
Keras = tf.keras
Sequential = Keras.models.Sequential
Activation = Keras.layers.Activation
Adam = Keras.optimizers.Adam
Dense, Dropout, BatchNormalization = Keras.layers.Dense, Keras.layers.Dropout, Keras.layers.BatchNormalization

def create_model(input_dim, config):
    learn_rate = config.get('learn_rate', 1e-3)
    dropout_rate = config.get('dropout_rate')
    batchnorm = config.get('batchnorm', False)
    hidden_layers = config.get('hidden_layers', [32])
    activation = config.get('activation', 'relu')
    optimizer = config.get('optimizer', Adam)

    model = Sequential()

    for idx, width in enumerate(hidden_layers):
        input_dim = input_dim if idx == 0 else None

        # hidden layers
        model.add(Dense(width, input_dim=input_dim))
        if batchnorm:
            model.add(BatchNormalization())
        model.add(Activation(activation))
        if dropout_rate != None:
            model.add(Dropout(dropout_rate))

    # output layer
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer(learn_rate))

    return model



class KerasRocAucCallback(Keras.callbacks.Callback):
    def __init__(self, frequency, x, y, withEarlyStop=False, logger=None):
        super(Keras.callbacks.Callback, self).__init__()
        self.frequency = frequency
        self.x = x
        self.y = y
        self.auc_scores = []
        self.epochs = []
        self.withEarlyStop = withEarlyStop
        self.logger = logger

    def on_epoch_end(self, epoch, logs, generator=None):
        logs = logs or {}
#         if logs.get('val_auc') == None:
#             logs['val_auc'] = [
        if epoch % self.frequency == 0:
            probs = self.model.predict(self.x)
            auc = roc_auc_score(self.y, probs)
            self.auc_scores.append(auc)
            logs['val_auc'] = auc
            self.epochs.append(f'ep{epoch}')
            
            if self.logger != None:
                self.logger.log_time(f'Epoch: {epoch}  AUC: {auc}').write_to_file()
            
            # check for early stopping criteria
            if not self.withEarlyStop or epoch < 15:
                return
            baseline_auc = self.auc_scores[-10]
            min_delta = 0.1
            last_max = np.max(self.auc_scores[-10])
            self.logger.log_time(f'Baseline {baseline_auc}, Max {last_max}').write_to_file()
            if last_max <= baseline_auc + min_delta:
                self.logger.log_time(f'Forcing stop at epoch {epoch}')
                self.model.stop_training = True
            if epoch > 20:
                self.logger.log_time(f'Forcing stop at epoch {epoch}')
                self.model.stop_training = True


    def get_aucs(self):
        return self.auc_scores
