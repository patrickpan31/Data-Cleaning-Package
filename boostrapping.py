#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler


# In[32]:


def boostrapping(X, Y, model, n, metric):
    
    
    if metric == 'importance':
        
        if isinstance(model, LogisticRegression):
            coefficient = np.zeros(X.shape[1])
            for i in range(n):
                scaler = StandardScaler()
                random_number = random.choices(range(len(X)), k = len(X))                            
                x_boost = X.iloc[random_number]
                y_boost = Y.iloc[random_number]
                x_std = scaler.fit(x_boost).transform(x_boost)
                logreg_mod.fit(x_boost, y_boost)
                coefficient = coefficient + logreg_mod.coef_

            ans = coefficient/n
            importance = pd.Series(np.abs(ans[0]), index = x_boost.columns)
            importance = importance.sort_values(ascending = False)


            return importance
        
        if isinstance(model, LinearRegression):
            coefficient = np.zeros(X.shape[1])
            for i in range(n):

                scaler = StandardScaler()
                random_number = random.choices(range(len(X)), k = len(X))                            
                x_boost = X.iloc[random_number]
                y_boost = Y.iloc[random_number]
                x_std = scaler.fit(x_boost).transform(x_boost)
                model.fit(x_boost, y_boost)
                coefficient = coefficient + model.coef_

            result = coefficient/n
            importance = pd.Series(np.abs(result), index = x_boost.columns)
            importance = importance.sort_values(ascending = False)

            return importance
    
    
    
    
    
    

