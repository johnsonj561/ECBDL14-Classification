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
TensorBoard = tf.keras.callbacks.TensorBoard

# ecbdl14_root = '/Users/jujohnson/git/ECBDL14-Classification'
ecbdl14_root = '/home/jjohn273/git/ECBDL14-Classification/'
sys.path.append(ecbdl14_root)
from model import create_model, write_model
from CustomCallbacks import KerasRocAucCallback

############################################
# Parse CLI Args
############################################
config = {}
cli_args = args_to_dict(sys.argv)
pos_size = float(cli_args.get('pos_size', 1))
neg_fraction = float(cli_args.get('neg_size', 1))
epochs = int(cli_args.get('epochs'))
runs = int(cli_args.get('runs'))


############################################
#  Create DNN Config
############################################
config = {
    'hidden_layers': [128, 128, 64, 32],
    'learn_rate': 0.001,
    'batch_size': 256,
    'dropout_rate': 0.2,
    'batchnorm': True
}


############################################
# Define I/O Paths
############################################
# inputs
data_path = os.path.join(ecbdl14_root, 'data/ecbdl14.onehot.sample.hdf')
data_key = 'train'
# outputs
now = datetime.datetime.today()
ts = now.strftime("%m%d%y-%H%M%S")
architecture_output = 'model-architecture.json'


############################################
# Initialize Logger
############################################
log_file = f'logs/{ts}.txt'
logger = Logger(log_file)
logger.log_time('Starting trainer')
logger.write_to_file()


############################################
# Iterate Over Runs
############################################
for run in runs:
    logger.log_time(f'Starting run {run}')).write_to_file()

    # Load Data
    df = pd.read_hdf(data_path, data_key)
    y, x = df['target'], df.drop(columns=['target'])
    del df

    # Sample Pos/Neg Classes Separately
    y_pos, y_neg = y.loc[y == 1], y.loc[y == 0]
    y_pos = y_pos.sample(frac=positive_fraction, replace=(positive_fraction > 1))
    y_neg = y_neg.sample(frac=negative_fraction, replace=(negative_fraction > 1))
    x_pos, x_neg = x.loc[y_pos.index], x.loc[y_neg.index]
    pos_count, neg_count = len(y_pos), len(y_neg)
    pos_ratio = pos_count / (pos_count + neg_count)
    x, y = pd.concat([x_pos, x_neg], copy=False), pd.concat([y_pos, y_neg], copy=False)
    del x_pos, x_neg, y_pos, y_nev

    # Log data stats
    logger.log_time('Data loaded')
    logger.log_time(f'Positive class size: {pos_count}')
    logger.log_time(f'Negative class size: {neg_count}')
    logger.log_time(f'Positive ratio: {pos_ratio}')

    # create model and save architecture to file if does not exist
    input_dim = x.shape[1]
    dnn = create_model(input_dim, config)
    if not os.path.isfile(architecture_output):
        write_model(dnn, architecture_output)
        print(f'Model architecture saved to {architecture_output}')

    # train model
    logger.log_time('Starting training...').write_to_file()
    dnn.fit(x_train, y_train, epochs=epochs, verbose=0)
    logger.log_time('Trainin complete!').write_to_file()

    # save weights
    out = f'{experiment}-{run}-model.h5'
    dnn.save_weights(out)
    logger.log_time(f'Model weights saved to {out}')

    # clean up data
    del x, y

logger.log_time('Job complete...').write_to_file()

