import pandas as pd 
import numpy as np
from LinearRegression import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations 


data = pd.read_csv("../data/standardized_data.csv",index_col="Unnamed: 0")
data = data.iloc[:,:-1]
combiner = combinations(data.columns,2)
# correlated_pairs = pd.read_csv("../data/correlated_pairs.csv")
# pairs = [(j["x"],j["y"]) for i,j in correlated_pairs.iloc[:,:-1].iterrows()]

def plot_linear_correlated(dataset):
    for x,y in list(combiner):
        df = dataset[[x,y]]
        model = LinearRegression(dataset=df,alpha=0.01,theta0=1,theta1=1)
        J,theta0,theta1 = model.stopping_criteria()
        df["prediction"] = df[x].apply(lambda x: (x*theta1) + theta0)
        SSE = sum((df[y]-df["prediction"])**2)
        SST = sum((df[y]-df[y].mean())**2)
        R2 = 1 - (SSE/SST)
        with open("./my_r2_output.txt","a") as f1:
            f1.write(f'{x},{y},{R2}\n')
        # ax = sns.scatterplot(data=df,x=x,y=y)
        # ax.set_title(f"{R2}")
        # ax.set_xlabel(f'{x}')
        # ax.set_ylabel(f'{y}')
        # plt.savefig(f'../q2/figures/{x}_{y}_scatterplot.png')
        # plt.clf()

plot_linear_correlated(dataset=data)
