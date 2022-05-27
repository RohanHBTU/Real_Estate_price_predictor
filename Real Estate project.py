#!/usr/bin/env python
# coding: utf-8

# ## Real Estate Project-Price Predictor

# In[1]:


import pandas as pd


# In[2]:


housing=pd.read_csv("housing.csv")


# In[3]:


housing.head()


# In[4]:


housing.describe()


# In[5]:


#Plotting histogram
import matplotlib.pyplot as plt
housing.hist(bins=30,figsize=(20,15))


# ## train-test splitting

# In[6]:


#Learning pratice of train/test data splitting

#import numpy as np
#def split_train_test(data,test_ratio):
#    np.random.seed(41)
#    shuffled=np.random.permutation(len(data))
#    print(shuffled)
#    test_set_size=int(len(data)*test_ratio)
#    test_indices=shuffled[:test_set_size]
#    train_indices=shuffled[test_set_size:]
#    return data.iloc[train_indices], data.iloc[test_indices]


# In[7]:


#train_set,test_set=split_train_test(housing,0.2)


# In[8]:


from sklearn.model_selection import train_test_split
train_set,test_set=train_test_split(housing,test_size=0.2,random_state=42)
print(f"rows in train set data: {len(train_set)}")


# In[9]:


from sklearn.model_selection import StratifiedShuffleSplit
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index, test_index in split.split(housing, housing[' CHAS']):
    strat_test_set=housing.loc[test_index]
    strat_train_set=housing.loc[train_index]


# In[10]:


housing=strat_train_set.copy()


# ## Establishing Corelation

# In[11]:


#corr_matrix=housing.corr()
#corr_matrix[" MEDV"].sort_values(ascending=False)


# In[12]:


from pandas.plotting import scatter_matrix
attributes=[" MEDV"," TAX"," LSTAT"]
scatter_matrix(housing[attributes],figsize=(12,8))


# In[13]:


housing.plot(kind="scatter",x=" RM",y=" MEDV",alpha=0.5)


# ## Attribute combination

# In[14]:


housing["TAXRM"]=housing[" TAX"]/housing[" RM"]


# In[15]:


corr_matrix=housing.corr()
corr_matrix[" MEDV"].sort_values(ascending=False)


# ## Separating features and labels in training dataframe

# In[16]:


housing=strat_train_set.drop(" MEDV",axis=1)
housing_labels=strat_train_set[" MEDV"].copy()


# ## Creating a Pipeline

# In[17]:


from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline=Pipeline([
    ('imputer',SimpleImputer(strategy="median")),
    ("std_scaler",StandardScaler())
])


# In[18]:


housing_num_tr=my_pipeline.fit_transform(housing)


# In[19]:


housing_num_tr


# ## Selecting a desired model 

# In[20]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#model=DecisionTreeRegressor()
#model=LinearRegression()
model=RandomForestRegressor()
model.fit(housing_num_tr,housing_labels)


# In[21]:


some_data=housing.iloc[:5]
some_labels=housing_labels.iloc[:5]
prepared_data=my_pipeline.transform(some_data)
model.predict(prepared_data)


# ## Evaluating the matrix

# In[22]:


import numpy as np
from sklearn.metrics import mean_squared_error
housing_predictions=model.predict(housing_num_tr)
lin_mse=mean_squared_error(housing_labels,housing_predictions)
lin_rmse=np.sqrt(lin_mse)
lin_mse


# ## Using better evaluation techniques - K Fold Cross Validation

# In[23]:


from sklearn.model_selection import cross_val_score
scores=cross_val_score(model, housing_num_tr, housing_labels,scoring='neg_mean_squared_error',cv=10)
rmse_scores=np.sqrt(-scores)


# In[24]:


#rsme_scores
scores


# In[25]:


def print_scores(scores):
    print("Scores:",scores)
    print("Mean:",scores.mean())
    print("Standard Deviation:",scores.std())


# In[26]:


print_scores(rmse_scores)


# ## Saving the model

# In[28]:


from joblib import dump,load
dump(model,'dragon.joblib')


# ## Testing the model on test data

# In[30]:


X_test=strat_test_set.drop(" MEDV",axis=1)
Y_test=strat_test_set[" MEDV"].copy()
X_test_prepared=my_pipeline.transform(X_test)
final_predictions=model.predict(X_test_prepared)
final_mse=mean_squared_error(Y_test,final_predictions)
final_rmse=np.sqrt(final_mse)
final_rmse


# In[32]:


X_test_prepared[0]


# In[ ]:




