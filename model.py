from sklearn.metrics import roc_auc_score
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
        self.frequency = frequency
        self.x = x
        self.y = y
        self.auc_scores = []
        self.epochs = []
        self.withEarlyStop = withEarlyStop
        self.logger = logger

    def on_epoch_end(self, epoch, logs, generator=None):
        if epoch % self.frequency == 0:
            probs = self.model.predict(self.x)
            auc = roc_auc_score(self.y, probs)
            self.auc_scores.append(auc)
            self.epochs.append(f'ep{epoch}')
            if self.logger != None:
                self.logger.log_time(f'Epoch: {epoch}\tAUC: {auc}').write_to_file()
            self.model.stop_training = should_stop(epoch)

    def should_stop(self, epoch):
        if not self.withEarlyStop or epoch < 20:
            return False
        baseline_auc = self.auc_scores[-10]
        min_delta = 0.002
        for auc in self.auc_scores[-9:]:
            if auc > (baseline_auc + min_delta):
                return False
        self.logger.log_time(f'Early stopping criteria met using a min_delta: {min_delta}')
        return True

    def get_aucs(self):
        return self.auc_scores
