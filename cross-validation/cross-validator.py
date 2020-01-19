import pandas as pd
import numpy as np
import os
import sys
import datetime
from sklearn.model_selection import StratifiedKFold, ParameterGrid
sys.path.append(os.environ['CMS_ROOT'])
from cms_modules.utils import model_summary_to_string, args_to_dict
from cms_modules.logging import Logger

ecbdl14_root = '~/git/ECBDL14-Classification/'
sys.path.append(ecbdl14_root)
from model import create_model, KerasAucCallback


#### Parse CLI Args

cli_args = args_to_dict(sys.argv)
hidden_layers = cli_args.get('hidden_layers')
hidden_layers = [int(nodes) for nodes in hidden_layers.split('|')]
learn_rate = float(cli_args.get('learn_rate', 1e-3))
batch_size = int(cli_args.get('batch_size', 128))
dropout_rate = cli_args.get('dropout_rate')
dropout_rate = float(dropout_rate) if dropout_rate != None
batchnorm = cli_args.get('batchnorm', 'false')
batchnorm = True if batchnorm.lower() == 'true'
debug = cli_args.get('debug', 'false')
debug = True if debug.lower() == 'true'
callback_freq = 5


#### Define I/O Paths

# inputs
data_path = os.path.join(ecbdl14_root, 'data/ecbdl14.onehot.sample.hdf')
data_key = 'train'

# outputs
now = datetime.datetime.today()
ts = now.strftime("%m%d%y-%H%M%S")
validation_auc_outputs = f'{ts}-validation-auc-results.csv'
train_auc_outputs = f'{ts}-train-auc-results.csv'

logger = Logger()
logger.log_time('Starting grid search job')
logger.log_message(f'Outputs being written to {[validation_auc_outputs,train_auc_outputs]}')


#### Initialize Output File Headers

config_value = f'layers:{hidden_layers_desc}-learn_rate:{learn_rate}-batch_size:{batch_size}-dropout_rate:{dropout_rate}-bathcnorm:{batchnorm}'

if !os.path.isfile(train_auc_outputs):
    results_header = 'config,fold,' + ','.join([f'ep_{i}' for i in range(epochs) if i%callback_freq == 0])
    output_files = [train_auc_outputs, validation_auc_outputs]
    output_headers = [results_header,results_header]
    for file, header in zip(output_files, output_headers):
        with open(file, 'w') as fout:
            fout.write(header + '\n')

def write_results(file, results):
    with open(file, 'a') as fout:
        fout.write(results + '\n')


#### Load Data

df = pd.read_hdf(data_path, data_key)
logger.log_time(f'Loaded data with shape {df.shape}')


#### Take Subset of Data In Debug

if debug:
    y, x = df[:10000]['target'], df[:10000].drop(columns=['target'])
else:
    y, x = df['target'], df.drop(columns=['target'])


#### Define Grid Search and Model Params
# Due to known issues with GridSearch and Keras callbacks, we enumerate grid options and manually iterate over each configuration.

stratified_cv = StratifiedKFold(n_splits=3, shuffle=True)

# ### Run Cross-Validation

logger.log_time('Starting cross-validation')

logger.log_message(f'Using config: {config_value}')

# iterate over cross-validation folds
for fold, (train_index, validate_index) in enumerate(stratified_cv.split(x, y)):
    logger.log_time(f'Starting fold {fold}')
    # prepare input data
    x_train, y_train = x.iloc[train_index].values, y.iloc[train_index].values
    x_valid, y_valid = x.iloc[validate_index].values, y.iloc[validate_index].values
    input_dim = x_train.shape[1]

    # setup callbacks to monitor auc
    validation_auc_callback = KerasAucCallback(callback_frequency, x_valid, y_valid)
    train_auc_callback = KerasAucCallback(callback_frequency, x_train, y_train)
    callbacks = [validation_auc_callback, train_auc_callback]

    # create model and log it's description on 1st run
    dnn = create_model(input_dim, config)
    if fold == 0:
        logger.log_message(f'Model summary')
        logger.log_message(model_summary_to_string(dnn))

    # train model
    logger.log_time('Starting training...')
    history = dnn.fit(x_train, y_train, epochs=epochs, callbacks=callbacks, verbose=0)
    logger.log_time('Trainin complete!')

    # write results
    prefix = f'{config_value},{fold}'
    validation_aucs = np.array(validation_auc_callback.get_aucs(), dtype=str)
    write_results(validation_auc_outputs, f'{prefix},{",".join(validation_aucs)}')
    train_aucs = np.array(train_auc_callback.get_aucs(), dtype=str)
    write_results(train_auc_outputs, f'{prefix},{",".join(train_aucs)}')


logger.log_time('Job complete...')

