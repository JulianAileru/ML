import pandas as pd 
import numpy as np
from LinearRegression import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("../data/standardized_data.csv")
data = data.iloc[:,:-1]
correlated_pairs = pd.read_csv("../data/correlated_pairs.csv")
pairs = [(j["x"],j["y"]) for i,j in correlated_pairs.iloc[:,:-1].iterrows()]

def plot_linear_correlated(dataset,pairs):
    for x,y in pairs:
        df = dataset[[x,y]]
        model = LinearRegression(dataset=df,alpha=0.01,theta0=1,theta1=1)
        J,theta0,theta1 = model.stopping_criteria()
        df["prediction"] = df[x].apply(lambda x: (x*theta1) + theta0)
        SSE = sum((df[y]-df["prediction"])**2)
        SST = sum((df[y]-df[y].mean())**2)
        R2 = 1 - (SSE/SST)
        print(f'{x},{y},{R2}')
        ax = sns.scatterplot(data=df,x=x,y=y)
        ax.set_title(f"{R2}")
        ax.set_xlabel(f'{x}')
        ax.set_ylabel(f'{y}')
        plt.savefig(f'../q2/figures/{x}_{y}_scatterplot.png')
        plt.clf()

plot_linear_correlated(dataset=data,pairs=pairs)
