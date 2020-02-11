import sys
print(f'Python version: {sys.version}')
import pandas as pd
import numpy as np
import os
import datetime
from sklearn.model_selection import StratifiedKFold, ParameterGrid
sys.path.append(os.environ['CMS_ROOT'])
from cms_modules.utils import model_summary_to_string, args_to_dict
from cms_modules.logging import Logger

import tensorflow as tf
TensorBoard = tf.keras.callbacks.TensorBoard

# ecbdl14_root = '/Users/jujohnson/git/ECBDL14-Classification'
ecbdl14_root = '/home/jjohn273/git/ECBDL14-Classification/'
sys.path.append(ecbdl14_root)
from model import create_model, write_model
from CustomCallbacks import KerasThresholdMonitoringCallback



############################################
# Parse CLI Args
############################################
config = {}
cli_args = args_to_dict(sys.argv)
pos_fraction = float(cli_args.get('positive_fraction', 1))
neg_fraction = float(cli_args.get('negative_fraction', 1))
epochs = int(cli_args.get('epochs'))
runs = int(cli_args.get('runs'))
pos_ratio = cli_args.get('positive_ratio')
debug = bool((cli_args.get('debug', 'false')))
positive_ratio = cli_args.get('positive_ratio')
gpus = int(cli_args.get('gpus', 0))

print(f'Loaded cli args: positive_ratio: {positive_ratio} pos_fraction:{pos_fraction} neg_fraction:{neg_fraction} epochs:{epochs} runs:{runs} debug: {debug}')



############################################
#  Create DNN Config
############################################
config = {
    'hidden_layers': [128, 128, 64, 32],
    'learn_rate': 0.001,
    'batch_size': 256,
    'dropout_rate': 0.2,
    'batchnorm': True,
    'gpus': gpus
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
weights_dir = 'weights'



############################################
# Initialize Logger
############################################
log_file = f'logs/{positive_ratio}-{ts}.txt'
logger = Logger(log_file)
logger.log_time('Starting trainer')
logger.write_to_file()


results = []

############################################
# Iterate Over Runs
############################################
for run in range(runs):
    logger.log_time(f'Starting run {run}').write_to_file()

    # Load Data
    df = pd.read_hdf(data_path, data_key)
    y, x = df['target'], df.drop(columns=['target'])
    del df

    # Sample Pos/Neg Classes Separately
    y_pos, y_neg = y.loc[y == 1], y.loc[y == 0]
    y_pos = y_pos.sample(frac=pos_fraction, replace=(pos_fraction > 1))
    y_neg = y_neg.sample(frac=neg_fraction, replace=(neg_fraction > 1))
    x_pos, x_neg = x.loc[y_pos.index], x.loc[y_neg.index]
    pos_count, neg_count = len(y_pos), len(y_neg)
    pos_ratio = pos_count / (pos_count + neg_count)
    x, y = pd.concat([x_pos, x_neg], copy=False), pd.concat([y_pos, y_neg], copy=False)
    del x_pos, x_neg, y_pos, y_neg

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

    # init callbacks
    cb = KerasThresholdMonitoringCallback(x, y, logger)
    tb = TensorBoard(log_dir='logs', histogram_freq=2)
    
    # train model
    logger.log_time('Starting training...').write_to_file()
   
    history = dnn.fit(x, y, epochs=epochs, verbose=1, callbacks=[cb,tb])
    logger.log_time('Trainin complete!').write_to_file()
    logger.log_time(f'History: {history.history}').write_to_file()
    results.append(history.history)

    # save weights
    model_output = f'{positive_ratio}-{run}-model.h5'
    dnn.save_weights(os.path.join(weights_dir, model_output))
    logger.log_time(f'Model weights saved to {model_output}')

    # clean up data
    del x, y

logger.log_time('Job complete...').write_to_file()
logger.log_time("\n".join(results))

