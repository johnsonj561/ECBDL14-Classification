import sys
print('Python version')
print(sys.version)
import pandas as pd
import numpy as np
import os
import datetime
from sklearn.model_selection import StratifiedKFold, ParameterGrid
sys.path.append(os.environ['CMS_ROOT'])
from cms_modules.utils import model_summary_to_string, args_to_dict
from cms_modules.logging import Logger

import tensorflow as tf
EarlyStopping = tf.keras.callbacks.EarlyStopping
ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau
TensorBoard = tf.keras.callbacks.TensorBoard

ecbdl14_root = '/home/jjohn273/git/ECBDL14-Classification/'
sys.path.append(ecbdl14_root)
from model import create_model
from CustomCallbacks import KerasRocAucCallback

############################################
# Parse CLI Args & Create DNN Config
############################################
config = {}
cli_args = args_to_dict(sys.argv)
hidden_layers_markup = cli_args.get('hidden_layers')
config['hidden_layers'] = [int(nodes) for nodes in hidden_layers_markup.split('+')]
config['learn_rate'] = float(cli_args.get('learn_rate', 1e-3))
config['batch_size'] = int(cli_args.get('batch_size', 128))
dropout_rate = cli_args.get('dropout_rate')
config['dropout_rate'] = float(dropout_rate) if dropout_rate != None else dropout_rate
batchnorm = cli_args.get('batchnorm', 'false')
config['batchnorm'] = True if batchnorm.lower() == 'true' else False
epochs = int(cli_args.get('epochs', 10))
debug = cli_args.get('debug', 'false')
debug = True if debug.lower() == 'true' else False
use_lr_reduction = cli_args.get('use_lr_reduction', 'false')
use_lr_reduction = True if use_lr_reduction.lower() == 'true' else False

############################################
# Define I/O Paths
############################################
# inputs
data_path = os.path.join(ecbdl14_root, 'data/ecbdl14.onehot.sample.hdf')
data_key = 'train'
# outputs
now = datetime.datetime.today()
ts = now.strftime("%m%d%y-%H%M%S")
validation_auc_outputs = f'{ts}-validation-auc-results.csv'
train_auc_outputs = f'{ts}-train-auc-results.csv'


############################################
# Initialize Output Files
############################################
config_value = f'layers:{hidden_layers_markup}-learn_rate:{config.get("learn_rate")}'
config_value += f'-batch_size:{config.get("batch_size")}-dropout_rate:{config.get("dropout_rate")}-bathcnorm:{config.get("batchnorm")}'

if not os.path.isfile(train_auc_outputs):
    results_header = 'config,fold,' + ','.join([f'ep_{i}' for i in range(epochs)])
    output_files = [train_auc_outputs, validation_auc_outputs]
    output_headers = [results_header,results_header]
    for file, header in zip(output_files, output_headers):
        with open(file, 'w') as fout:
            fout.write(header + '\n')

def write_results(file, results):
    with open(file, 'a') as fout:
        fout.write(results + '\n')

############################################
# Initialize Logger
############################################
tensorboard_dir = f'tensorboard/{ts}-{config_value}/'
log_file = f'logs/{ts}-{config_value}.txt'
logger = Logger(log_file)
logger.log_time('Starting grid search job')
logger.log_time(f'Outputs being written to {[validation_auc_outputs,train_auc_outputs]}')
logger.write_to_file()


############################################
# Load Data
############################################
df = pd.read_hdf(data_path, data_key)
logger.log_time(f'Loaded data with shape {df.shape}').write_to_file()
if debug:
    y, x = df[:10000]['target'], df[:10000].drop(columns=['target'])
else:
    y, x = df['target'], df.drop(columns=['target'])

del df

# Iterate Over K-Fold Validation
############################################
stratified_cv = StratifiedKFold(n_splits=3, shuffle=True)
logger.log_time('Starting cross-validation')
logger.log_time(f'Using config: {config_value}')

# iterate over cross-validation folds
for fold, (train_index, validate_index) in enumerate(stratified_cv.split(x, y)):
    logger.log_time(f'Starting fold {fold}').write_to_file()
    # prepare input data
    x_train, y_train = x.iloc[train_index].values, y.iloc[train_index].values
    x_valid, y_valid = x.iloc[validate_index].values, y.iloc[validate_index].values
    input_dim = x_train.shape[1]
    del train_index, validate_index

    # setup callbacks for monitoring AUC and early stopping
    validation_auc_callback = KerasRocAucCallback(x_valid, y_valid, True, logger)
    train_auc_callback = KerasRocAucCallback(x_train, y_train)
    early_stopping = EarlyStopping(monitor='val_auc', min_delta=0.001, patience=15, mode='max')
    tensorboard = TensorBoard(log_dir=f'{tensorboard_dir}/fold-{fold}', write_graph=False)
    callbacks = [validation_auc_callback, train_auc_callback, early_stopping, tensorboard]
    if use_lr_reduction:
        callbacks.append(ReduceLROnPlateau(
            monitor='val_auc', factor=0.1, patience=5,
            mode='max', cooldown=0, min_lr=0.0001))

    # create model and log it's description on 1st run
    dnn = create_model(input_dim, config)
    if fold == 0:
        logger.log_time(model_summary_to_string(dnn)).write_to_file()

    # train model
    logger.log_time('Starting training...').write_to_file()
    history = dnn.fit(x_train, y_train, epochs=epochs, callbacks=callbacks, verbose=0)
    logger.log_time('Trainin complete!').write_to_file()

    # write results
    prefix = f'{config_value},{fold}'
    validation_aucs = np.array(history.history['val_auc'], dtype=str)
    write_results(validation_auc_outputs, f'{prefix},{",".join(validation_aucs)}')
    train_aucs = np.array(history.history['train_auc'], dtype=str)
    write_results(train_auc_outputs, f'{prefix},{",".join(train_aucs)}')

    # free some memory
    del history, x_valid, y_valid, x_train, y_train
    del dnn


logger.log_time('Job complete...').write_to_file()

