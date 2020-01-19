#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd
import numpy as np
import os
import sys
import datetime
sys.path.append(os.environ['CMS_ROOT'])
from cms_modules.utils import model_summary_to_string
from cms_modules.logging import Logger


# In[ ]:


ecbdl14_root = '/home/jjohn273/git/ECBDL14-Classification/'
sys.path.append(ecbdl14_root)
from model import create_model, KerasAucCallback


# In[31]:


debug = False


# ### Define I/O Paths

# In[18]:


# inputs
data_path = os.path.join(ecbdl14_root, 'data/ecbdl14.onehot.sample.hdf')
data_key = 'train'

# outputs
now = datetime.datetime.today()
ts = now.strftime("%m%d%y-%H%M%S")
config_outputs = f'{ts}-configs.csv'
validation_auc_outputs = f'{ts}-validation-auc-results.csv'
train_auc_outputs = f'{ts}-train-auc-results.csv'

logger = Logger()
logger.log_time('Starting grid search job')
logger.log_message(f'Outputs being written to {[config_outputs,validation_auc_outputs,train_auc_outputs]}')


# ### Load Data

# In[19]:


df = pd.read_hdf(data_path, data_key)
logger.log_time(f'Loaded data with shape {df.shape}')


# ### Take Subset of Data In Debug

# In[20]:


if debug:
    y, x = df[:10000]['target'], df[:10000].drop(columns=['target'])
else:
    y, x = df['target'], df.drop(columns=['target'])


# ### Define Grid Search and Model Params
# 
# Due to known issues with GridSearch and Keras callbacks, we enumerate grid options and manually iterate over each configuration.

# In[21]:


from sklearn.model_selection import StratifiedKFold, ParameterGrid

stratified_cv = StratifiedKFold(n_splits=3, shuffle=True)

hidden_layers = [[32,32],[64,64],[128,128],[128,64],[32,32,32],[64,64,64],[128,64,32,16]]

param_grid = dict(
  hidden_layers=hidden_layers,
  learn_rate=[1e-3],
  batch_size=[128],
  dropout_rate=[None, 0.5],
  batchnorm=[True, False])

epochs = 300
score_freq = 5

param_grid_options = list(ParameterGrid(param_grid))

logger.log_message('Set up grid search parameters:')
for option in param_grid_options:
    logger.log_message(f'{option}')


# ### Write Headers to Output Files

# In[22]:


# initialize output csv headers
config_header = 'config,hidden_layers,learn_rate,dropout,batchnorm'
results_header = 'config,fold,' + ','.join([f'ep_{i}' for i in range(epochs) if i%score_freq == 0])
output_files = [config_outputs, train_auc_outputs, validation_auc_outputs]
output_headers = [config_header,results_header,results_header]

for file, header in zip(output_files, output_headers):
    with open(file, 'w') as fout:
        fout.write(header + '\n')
        
def write_results(file, results):
    with open(file, 'a') as fout:
        fout.write(results + '\n')


# ### Run Cross-Validation

# In[29]:


logger.log_time('Starting cross-validation')

# iterate over grid options and write results
for config_idx, config in enumerate(param_grid_options):
    # set up model config
    learn_rate = config.get('learn_rate')
    dropout_rate = config.get('dropout_rate')
    batchnorm = config.get('batchnorm')
    hidden_layers = config.get('hidden_layers')
    
    hidden_layers_desc = "|".join(np.array(hidden_layers, dtype=str))
    config_str = f'{config_idx},layers:{hidden_layers_desc},learn_rate:{learn_rate},dropout:{dropout_rate},batchnorm:{batchnorm}'
    write_results(config_outputs, config_str)
    logger.log_message(f'Using config: {config_idx}\n{config}')
    
    # iterate over cross-validation folds
    for fold, (train_index, validate_index) in enumerate(stratified_cv.split(x, y)):
        logger.log_time(f'Starting fold {fold} for config {config_idx}')
        # prepare input data
        x_train, y_train = x.iloc[train_index].values, y.iloc[train_index].values
        x_valid, y_valid = x.iloc[validate_index].values, y.iloc[validate_index].values
        input_dim = x_train.shape[1]
        
        # setup callbacks to monitor auc
        score_frequency = 2
        config['fold'] = fold + 1
        validation_auc_callback = KerasAucCallback(score_frequency, x_valid, y_valid)
        train_auc_callback = KerasAucCallback(score_frequency, x_train, y_train)
        callbacks = [validation_auc_callback, train_auc_callback]
        
        # create model
        dnn = create_model(input_dim, config)
        # log stats if first fold
        if fold == 0:
            logger.log_message(f'Training fold shape {x_train.shape}\nValidation fold shape {x_valid.shape}')
            logger.log_message(f'Model summary for configuration: {config_str}')
            logger.log_message(model_summary_to_string(dnn))
        
        # train model
        logger.log_time('Starting training...')
        history = dnn.fit(x_train, y_train, epochs=epochs, callbacks=callbacks, verbose=0)
        logger.log_time('Trainin complete!')
        
        # write results
        prefix = f'{config_idx},{fold}'
        validation_aucs = np.array(validation_auc_callback.get_aucs(), dtype=str)
        write_results(validation_auc_outputs, f'{prefix},{",".join(validation_aucs)}')
        train_aucs = np.array(train_auc_callback.get_aucs(), dtype=str)
        write_results(train_auc_outputs, f'{prefix},{",".join(train_aucs)}')


# In[ ]:


logger.log_time('Job complete...')

