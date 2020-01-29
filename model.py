import tensorflow as tf
Keras = tf.keras
Sequential = Keras.models.Sequential
Activation = Keras.layers.Activation
Adam = Keras.optimizers.Adam
Dense, Dropout, BatchNormalization = Keras.layers.Dense, Keras.layers.Dropout, Keras.layers.BatchNormalization
multi_gpu_model = Keras.utils.multi_gpu_model
K =  Keras.backend

def create_model(input_dim, config):
    K.clear_session()
    learn_rate = config.get('learn_rate', 1e-3)
    dropout_rate = config.get('dropout_rate')
    batchnorm = config.get('batchnorm', False)
    hidden_layers = config.get('hidden_layers', [32])
    activation = config.get('activation', 'relu')
    optimizer = config.get('optimizer', Adam)    
    gpu_count = config.get('gpus', 0)

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
    
    # use gpus?
    if gpu_count > 1:
        model = multi_gpu_model(model, gpu_count)

    model.compile(loss='binary_crossentropy', optimizer=optimizer(learn_rate))

    return model


def write_model(model, path):
    json = model.to_json()
    with open(path, 'w') as out:
        out.write(json)

