#!/usr/bin/env python
# coding: utf-8

# In[2]:


from joblib import dump,load
import numpy as np
model=load('dragon.joblib')


# In[5]:


features=np.array([[-0.44228927, -0.4898311 , -1.37640684, -0.27288841, -0.34321545,
        0.36524574, -0.33092752,  1.20235683, -1.0016859 ,  0.05733231,
       -1.21003475,  0.38110555, -0.57309194],])
print(model.predict(features))


# In[ ]:




