#!/usr/bin/env python
# coding: utf-8

# In[100]:


import numpy as np
import pandas as pd
import random


# In[101]:


def sigmoid(z):
    
    return 1/(1+np.exp(-z))


# In[134]:


def cal_weight(Y):
    
    weight = np.zeros(len(Y))
    n_0 = max(np.sum(Y), len(Y)-np.sum(Y))
    n_1 = min(np.sum(Y), np.abs(len(Y)-np.sum(Y)))
    N = len(Y)
    for i in range(len(Y)):
        weight[i] =  (N / (n_0 * (1 - Y[i]) + n_1 * Y[i]))
        
    return weight/(np.sum(weight))


# In[205]:


def loss_function(X,Y,W, B, reg = 0):
    
    ### x(m,)
    ### y(1,)
    ### w: weight of x
    M = X.shape[0]
    cost = 0
    weight = cal_weight(Y)
    
    weight = np.zeros(len(Y))
    for i in range(len(Y)):
        weight[i] = 1
    
    for i in range(M):
        
        fx = np.dot(W,X[i])+B
        cost += (-1*Y[i]*weight[i]*np.log(sigmoid(fx)) - (1-y[i])*weight[i]*np.log(1-sigmoid(fx)))
        
    return cost/M


# In[206]:


def gradient(X,Y,W,B,reg = 0):
    
    w_gradient = np.zeros(X.shape[1])
    b_gradient = 0
    
    M,N = X.shape
    weight = cal_weight(Y)
    
    weight = np.zeros(len(Y))
    for i in range(len(Y)):
        weight[i] = 1
        
    for i in range(M):
        
        z = np.dot(X[i],W) + B
        fx = sigmoid(z)
        
        for j in range(N):
            w_gradient[j] += weight[i]*(fx - Y[i])*X[i][j]
        b_gradient += weight[i]*(fx - Y[i])
    w_gradient /= M
    b_gradient /= M
    
    return w_gradient, b_gradient
    
    


# In[207]:


def gradient_decent(X,Y,W,B, iteration, lambda_, lossfunction, gradientfunction, reg):
    
    Total_cost = []
    M,N = X.shape
    w_init = W
    b_init = B
    for i in range(iteration):
        dj_dw, dj_db = gradientfunction(X,Y, w_init, b_init,reg)
        
        w_init = w_init - lambda_ * dj_dw
        b_init = b_init - lambda_ * dj_db
        
        cost = lossfunction(X,Y,w_init,b_init,reg)
        Total_cost.append(cost)
        
        if (i/100)%1 == 0:
            print(f"Iteration {i:4}: Cost {float(cost):8.2f}   ")
        
    return w_init, b_init, cost


# In[237]:


#w,b, J_history = gradient_decent(x ,y, initial_w, initial_b, iterations,alpha,
                                   #loss_function, gradient, 0)


# In[106]:


gradient_decent(X,y,initial_w,initial_b,weight, iterations, alpha, reg_cost, regularization)


# In[117]:


def reg_cost(X,Y,W,B,reg):
    
    total_lost = loss_function(X,Y,W,B)
    
    rl = 0
    for i in range(X.shape[1]):
        rl += (reg*(W[i]**2))/(2*len(X))
        
    total_lost = total_lost + rl
    return total_lost


# In[118]:


def regularization(X,Y,W,B,reg):
    
    w_gradient, b_gradient = gradient(X,Y,W,B)
    
    for i in range(X.shape[1]):
        w_gradient[i] += (reg/m)*(W[i])
        
    return w_gradient, b_gradient


# In[ ]:


def model_fit(X,Y, iteration, alpha, regulation):
    
    initial_w = np.random.rand(X.shape[1])-0.5
    initial_b = 1.
    
    w_final, b_final,cost = gradient_decent(X,Y,initial_w,initial_b, iteration, alpha, reg_cost, regularization, regulation)
    
    


# In[371]:


class weighted_logi(object):
    
    def __init__(self):
        
        self.w = 0.5
        self.b = 1.
        
    
    def sigmoid(self,z):
    
        return 1/(1+np.exp(-z))
    
    def cal_weight(self,Y):
    
        weight = np.zeros(len(Y))
        n_0 = max(sum(Y), len(Y)-sum(Y))
        n_1 = min(sum(Y), np.abs(len(Y)-sum(Y)))
        N = len(Y)
        for i in range(len(Y)):
            weight[i] =  (N / (n_0 * (1 - Y[i]) + n_1 * Y[i]))

        return weight/(sum(weight))
    
    def loss_function(self,X,Y,W, B, reg = 0):
    
        ### x(m,)
        ### y(1,)
        ### w: weight of x
        M = len(X)
        cost = 0
        weight = self.cal_weight(Y)

        weight = np.zeros(len(Y))
        for i in range(len(Y)):
            weight[i] = 1

        for i in range(M):

            fx = np.dot(W,X.iloc[i])+B
            cost += (-1*Y[i]*weight[i]*np.log(sigmoid(fx)) - (1-y[i])*weight[i]*np.log(1-sigmoid(fx)))

        return cost/M
    
    
    def gradient(self,X,Y,W,B,reg = 0):
    
        w_gradient = np.random.rand(X.shape[1])-0.5
        b_gradient = 0

        M,N = len(X), len(X.iloc[0])
        weight = self.cal_weight(Y)

        weight = np.zeros(len(Y))
        for i in range(len(Y)):
            weight[i] = 1

        for i in range(M):

            z = np.dot(X.iloc[i],W) + B
            fx = self.sigmoid(z)

            for j in range(N):
                w_gradient[j] += weight[i]*(fx - Y[i])*X.iloc[i][j]
            b_gradient += weight[i]*(fx - Y[i])
        w_gradient /= M
        b_gradient /= M

        return w_gradient, b_gradient
    
    
    def reg_cost(self,X,Y,W,B,reg):
    
        total_lost = self.loss_function(X,Y,W,B)

        rl = 0
        for i in range(len(X.iloc[0])):
            rl += (reg*(W[i]**2))/(2*len(X))

        total_lost = total_lost + rl
        return total_lost
    
    
    def regularization(self,X,Y,W,B,reg):
    
        w_gradient, b_gradient = self.gradient(X,Y,W,B)

        for i in range(len(X.iloc[0])):
            w_gradient[i] += (reg/m)*(W[i])

        return w_gradient, b_gradient
    
    
    def gradient_decent(self,X,Y,W,B, iteration, lambda_, lossfunction, gradientfunction, reg):
    
        Total_cost = []
        M,N = X.shape
        w_init = W
        b_init = B
        for i in range(iteration):
            
            dj_dw, dj_db = gradientfunction(X,Y, w_init, b_init,reg)

            w_init = w_init - lambda_ * dj_dw
            b_init = b_init - lambda_ * dj_db

            cost = lossfunction(X,Y,w_init,b_init,reg)
            Total_cost.append(cost)

            if (i/100)%1 == 0:
                print(f"Iteration {i:4}: Cost {float(cost):8.2f}   ")

        return w_init, b_init, cost
    
    
    
    def fit(self,X,Y, iteration = 10000, alpha = 0.2, regulation = None):
        
        if regulation is None:
            regulation = 0
        self.w = np.random.rand(len(X.iloc[0]))-0.2
        self.b = 1.

        self.w, self.b, cost = self.gradient_decent(X,Y,self.w,self.b, iteration, alpha, self.reg_cost, self.regularization, regulation)
        
    
    def predict_prob(self,X):
        
        result = np.zeros(len(X))
        
        for i in range(len(X)):
            
            z = np.dot(X.iloc[i], self.w) + self.b
            fx = self.sigmoid(z)
            
            result[i] = (fx)
        
        return result
            


        


# In[407]:


# n = 100
# x1 = np.random.normal(0, 1, n)
# x2 = np.random.normal(0, 1, n)
# y = (x1 + x2 > 0).astype(int)

# # Create a DataFrame with the data
# data = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})


# In[408]:


# y = data['y']
# x = data.drop('y',axis = 1)


# In[ ]:





# In[ ]:





# In[409]:


# weighted = weighted_logi()


# In[410]:


# weighted.fit(x,y)


# In[411]:


# np.sum((weighted.predict_prob(x)>0.5).astype(int) == y)


# In[412]:


# (weighted.predict_prob(x)>0.9).astype(int) == y


# In[413]:


# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import LogisticRegression


# In[414]:


# test = LogisticRegression()


# In[415]:


# test.fit(x,y)


# In[416]:


# test.coef_


# In[417]:


# weighted.w


# In[418]:


# test.predict_proba(x)[:,1]


# In[419]:


# weighted.predict_prob(x)


# In[420]:


# test = np.zeros((2,1)) + 1


# In[421]:


# test.shape


# In[422]:


# x


# In[ ]:





# In[ ]:




