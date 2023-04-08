#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import pandas as pd
import random
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.model_selection import train_test_split


# In[49]:


def random_unique_generator(n,m):
    
    ##### n: Range of number want to generate
    ##### m: List of number that we do not want in the return
    
    new_range = set(set(range(n))-set(m))
    result = random.sample(new_range, min(n,len(new_range)))
    
    return result


# In[50]:


def FPS(y_true,y_pre):
    
    ##### This function is used to calculate the False Positive Rate
    
    FP = np.sum((y_true == 0)&(y_pre==1))
    TN = np.sum((y_true == 0)&(y_pre==0))
    
    if (FP+TN) == 0:
        FP += 0.0000001
        TN += 0.0000001
    
    return FP/(FP+TN)


# In[51]:


def TPR(y_true,y_pre):
    
    ##### This function is used to calculate the True Positive Rate
    
    TP = np.sum((y_true == 1)&(y_pre==1))
    FN = np.sum((y_true == 1)&(y_pre==0))
    
    if (FP+TN) == 0:
        FP += 0.0000001
        TN += 0.0000001
    
    return TP/(TP+FN)


# In[52]:


def calculate_auc(tpr, fpr):
    # Sort TPR and FPR in ascending order of FPR
    sort_indices = np.argsort(fpr)
    tpr = tpr[sort_indices]
    fpr = fpr[sort_indices]

    # Calculate AUC using trapezoidal rule
    auc = np.trapz(tpr, fpr)

    return auc


# In[68]:


def auc_score(y_true, y_prob):
    
    fps = []
    tpr = []
    rng = np.arange(0, 1.01, 0.01)
    for i in range(rng):
        decision = (y_prob > i).astype(int)
        fps.append(FPS(y_true,decision))
        tpr.append(TPR(y_true,decision))
        
    fps = np.array(fps)
    tpr = np.array(tpr)
        
    return calculate_auc(tpr,fps)


# In[67]:


def roc(y_true, y_prob):
    
    fps = []
    tpr = []
    rng = np.arange(0, 1.01, 0.01)
    for i in (rng):
        decision = (y_prob > i).astype(int)
        fps.append(FPS(y_true,decision))
        tpr.append(TPR(y_true,decision))
        
    fps = np.array(fps)
    tpr = np.array(tpr)
        
    sort_indices = np.argsort(fps)
    tpr = tpr[sort_indices]
    fps = fps[sort_indices]
    
    score = calculate_auc(tpr, fps)
    
    plt.plot(fps, tpr, label='ROC curve (AUC = %0.2f)' % score)

    # Add labels and a title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")

    # Display the plot
    plt.show()  


# In[54]:


def precision(y_true, y_pre):
    
    TP = np.sum((y_true == 1)&(y_pre==1))
    TOTPRED = np.sum(y_pre)
    if TOTPRED == 0:
        TOTPRED += 0.00001
    
    return TP/TOTPRED


# In[55]:


def recall(y_true,y_pre):
    
    TP = np.sum((y_true == 1)&(y_pre==1))
    TOTP = np.sum(y_true)
    if TOTP == 0:
        TOTP += 0.00001
    
    return TP/TOTP


# In[56]:


def F1(y_true,y_pre):
    
    pres = precision(y_true,y_pre)
    rec = recall(y_true,y_pre)
    
    if pres == 0 and rec == 0:
        pres += 0.00001
        rec += 0.00001
    
    return 2*(pres * rec)/ (pres + rec)


# In[57]:


def cross_evaluation(X,Y,folds,model, error_type):
    
    number = random.sample(range(len(X)),len(X))
    num_fold = [[]for i in range(folds)]
    len_per_fold = len(X)//folds
    
    for i in range(folds):
        num_fold[i] = number[i*len_per_fold:(i+1)*len_per_fold]
    error = np.zeros(folds)
    
    for i in range(folds):
        index = []
        for j in range(folds):
            if j!=i:
                index += num_fold[j]
        train_x = X.iloc[index]
        train_y = Y.iloc[index]
        test_x = X.iloc[num_fold[i]]
        test_y = Y.iloc[num_fold[i]]
        model.fit(train_x,train_y)
        prediction = model.predict(test_x)
        
        if error_type == 'mse':
            err = np.sum((prediction-test_y)**2)
            error[i] = err
            
        if error_type == 'r2':
            SSRE = np.sum((prediction-test_y)**2)
            
            y_avg = np.mean(test_y)
            SStot = np.sum((test_y - y_avg)**2)

            R_Squre = 1-(SSRE/SStot)# * (len(train_x) - 1) / (len(train_x) - train_x.shape[1] - 1)

            error[i] = R_Squre
            
        if error_type == 'roc-auc':
            prob = model.predict_proba(x_test)[:,1]
            return auc_score(y_test, prob)

    return error
        
        
    
    


# In[58]:


def upsampling(X,Y,folds,model,error_type):
    
    y_index = Y.index[Y == 1].tolist()  #### find the index of y==1
    y_random = random.sample(y_index, len(y_index)) #### generate random index for y==1
    
    y_per_group = len(y_index)//folds
    if y_per_group == 0:
        print('Please try smaller number of folds')
        return
    target_y = X.iloc[y_index]          #### find the rows with y==1
    X_pure = X.drop(y_index)            #### Remove y==1 in original dataset
    
    number = random_unique_generator(len(X),y_index)
    num_folds = [[]for i in range(folds)]
    len_per_fold = len(X_pure)//folds
    
    y_folds = [[]for i in range(folds)] #### stored with y==1 in which fold
    
    for i in range(folds):
        num_folds[i] = number[i*len_per_fold:(i+1)*len_per_fold]
        y_folds[i] = y_random[i*y_per_group:(i+1)*y_per_group]
        
    ratio = len_per_fold//y_per_group #### number of time to upsample
    error = np.zeros(folds)
        
    for i in range(folds):
        index = []
        replicated_x = X.iloc[0:0]
        replicated_y = Y.iloc[0:0]
        for j in range(folds):
            if i != j:
                index += num_folds[j]
                replicated_x = pd.concat([replicated_x, pd.concat([X.iloc[y_folds[j]]]*ratio)], axis = 0)
                replicated_y = pd.concat([replicated_y, pd.concat([Y.iloc[y_folds[j]]]*ratio)],axis = 0)
        x_train = pd.concat([X.iloc[index], replicated_x], axis = 0)
        y_train = pd.concat([Y.iloc[index], replicated_y], axis = 0)
        x_test = pd.concat([X.iloc[num_folds[i]], pd.concat([X.iloc[y_folds[j]]]*ratio)],axis = 0)
        y_test = pd.concat([Y.iloc[num_folds[i]], pd.concat([Y.iloc[y_folds[j]]]*ratio)], axis = 0)
    
        model.fit(x_train,y_train)
        prediction = model.predict(x_test)
        
        if error_type == 'misclassification':

            err = np.sum(prediction == y_test)/len(y_test)
            error[i] = err
            
        if error_type == 'roc-auc':
            prob = model.predict_proba(x_test)[:,1]
            return auc_score(y_test, prob)
                
    
    return error
    

