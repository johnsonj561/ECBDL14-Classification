
# coding: utf-8

# In[581]:


import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import time


# In[432]:


debug = False


# ### Load Data

# In[473]:


col_file = open('/home/jjohn273/ecbdl14/columns.csv', 'r')
columns = col_file.read().strip().split(',')
columns = [col if col != 'class' else 'target' for col in columns]


# In[ ]:


df = pd.read_csv('~/ecbdl14/ecbdl14-train-shuf-sample.csv.gz', header=None, low_memory=False)
df.columns = columns
print(f'Loaded ecbdl14 data with shape {df.shape}')

# In[584]:


# in debug mode, use just a sample!
if debug:
    print('In debug mode, using subset of data')
    df = df[:int(50e3)]


# ### Check For Missing Data

# In[478]:


print(f'Are there missing values ? {df.isna().any().any()}')


# ### One Hot Encode Categorical Predictors

# In[479]:

print('One hot encoding categorical variables')
df = pd.get_dummies(df)


# ### Make Train/Test Split

# In[480]:


from sklearn.model_selection import train_test_split


# In[481]:

print('Making train/test split')
train, test = train_test_split(df, test_size=0.2, random_state=42)
print(f'Train shape {train.shape}', f'Test shape {test.shape}')


# ### Normalize + Feature Selection

# In[482]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold, chi2, SelectKBest
from sklearn.pipeline import Pipeline


# In[483]:


train_y, train_x = train.loc[:, ['target']], train.drop(columns=['target'])
test_y, test_x = test.loc[:, ['target']], test.drop(columns=['target'])


# In[484]:

print('Beginning feature selection')
start = time.time()

feature_selector = SelectKBest(chi2, k=200)

pipeline = Pipeline([
  ('normalize', MinMaxScaler()),
  ('strip_zero_variance', VarianceThreshold()),
  ('feature_selector', feature_selector)])

train_x_normalized = pipeline.fit_transform(train_x, train_y)
test_x_normalized = pipeline.transform(test_x)

columns = train_x.columns[feature_selector.get_support()]

train_x_normalized = pd.DataFrame(train_x_normalized, columns=columns, index=train_x.index)
test_x_normalized = pd.DataFrame(test_x_normalized, columns=columns, index=test_x.index)

train_normalized = pd.concat([train_x_normalized, train_y], axis=1)
test_normalized = pd.concat([test_x_normalized, test_y], axis=1)

end = time.time()

print(f'Feature selection completed in {end - start}')

# In[ ]:


train_normalized.to_hdf('ecbdl14.onehot.sample.hdf', 'train')
test_normalized.to_hdf('ecbdl14.onehot.sample.hdf', 'test')

print('Results written to hdf...\nJob complete')
