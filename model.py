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



class KerasAucCallback(Keras.callbacks.Callback):
    def __init__(self, frequency, x, y):
        self.frequency = frequency
        self.x = x
        self.y = y
        self.auc_scores = []
        self.epochs = []
        

    def on_epoch_end(self, epoch, logs, generator=None):
        if epoch % self.frequency == 0:
            probs = self.model.predict(self.x)
            self.auc_scores.append(roc_auc_score(self.y, probs))
            self.epochs.append(f'ep{epoch}')

    def get_epochs(self):
        return self.epochs
    
    def get_aucs(self):
        return self.auc_scores
