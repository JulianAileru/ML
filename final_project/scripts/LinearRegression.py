#!/usr/bin/env python
# coding: utf-8

import numpy as np 
import pandas as pd
import random
import matplotlib.pyplot as plt
import warnings 

warnings.filterwarnings('ignore')

class LinearRegression:
    import numpy as np 
    import pandas as pd 
    @classmethod 
    def OLS(cls,df,x_name='x',y_name='y'):
        if isinstance(df,pd.DataFrame):
            x_bar = np.mean(df[x_name])
            y_bar = np.mean(df[y_name])
            cov_matrix = np.cov(df[x_name],df[y_name]) #np.cov = covariance matrix (var_X, Cov(X,Y) | var_Y, Cov(Y,X))
            var_X = cov_matrix[0,0]
            cov_XY = cov_matrix[0,1]
            B_1 = (cov_XY/ var_X)
            B_0 = y_bar - (B_1*x_bar)
            df["y_estimated"] = df[x_name].apply(lambda x: B_0 + (B_1*x)) 
            df["residual"] = (df[y_name] - df["y_estimated"])
            #residual = sum(df["residual"])
        return [(B_0,B_1),df] 

            
    @classmethod 
    def normalization(cls,df):
        if isinstance(df,pd.DataFrame):
            for i in df.columns:
                mu = np.mean(df[i]) 
                df[i] = df[i].apply(lambda x: (x - mu) / np.std(df[i]))  
        cls.df = df  
        return cls.df 
    def __init__(self,dataset=None,alpha=None,theta0=None,theta1=None):
        self.dataset = dataset 
        self.alpha = alpha
        self.theta0 = theta0
        self.theta1 = theta1 
    def descent(self):
        dJdt0 = 0
        dJdt1 = 0
        m = len(self.dataset)
        for index,kvp in self.dataset.iterrows(): #kvp contains x and y values
            dJdt0 += ((self.theta0 + self.theta1*(kvp[0])) - kvp[1]) 
            dJdt1 += ((self.theta0 + self.theta1*(kvp[0])) - kvp[1]) * kvp[0] 
        self.theta0 = self.theta0 - (self.alpha*((dJdt0/m)))
        self.theta1 = self.theta1 - (self.alpha*((dJdt1/m)))
        return self.theta0,self.theta1
    def J(self):
        J = 0
        m = len(self.dataset)
        for index,kvp in self.dataset.iterrows():
            prediction = self.theta0 + (self.theta1*kvp[0])
            J += (1/(2*m))*(prediction - kvp[1])**2
        return J
    def iterations(self,n=100):
        cost_list = []
        for i in range(n):
            self.descent()
            cost = self.J()
            cost_list.append((cost,self.theta0,self.theta1))
        return sorted(cost_list,key=lambda x: x[0])
    def stopping_criteria(self,dJgT=True,threshold=0.00001,max_iterations=1000):
        cost_list = []
        deltaJ = 0
        iteration = 0 
        while dJgT and (iteration < max_iterations):
            self.descent()
            cost = self.J()
            cost_list.append((cost,self.theta0,self.theta1))
            if len(cost_list) > 1:
                deltaJ = abs(cost_list[-1][0] - cost_list[-2][0])
                if deltaJ < threshold:
                    dJgT = False
            iteration += 1
            if iteration > max_iterations:
                raise ValueError("max number of iterations reached")
        print(f'iteration_n:{iteration}')
        return sorted(cost_list,key=lambda x:x[0])[0]