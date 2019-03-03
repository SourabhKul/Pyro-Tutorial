"""
Intro to Probabilistic Programming: Bayesian Data Analysis with Bayesian Regression

by Sourabh Kulkarni (https://www.github.com/SourabhKul)

Following instructions from MLTrain@UAI 2018 Pyro Workshop (http://pyro.ai/examples/bayesian_regression.html)

Some basics before we get started:

Problem under consideration:
Analyzing effect of terrain ruggedness on GDP of african nations, adapted 
from Nunn, N. & Puga, D., Ruggedness: The blessing of bad geography in Africa”,
Review of Economics and Statistics 94(1), Feb. 2012
- Typically, terrain ruggedness is negetively correlated to GDP of a country
- In Africa, due to it's negetive effect on slavery, it may have a positive
  correlation to country's GDP
- In this code, we test this hypothesis using Bayesian Regression analysis  

To perform Bayesian Regression:
- we define the regression model in a stochastic function
- we declare the parameters to be learnt as being random variables with some prior
- we perform both SVI and MCMC for learning these parameters
- we discuss the tradeoffs involved among the two, and some nuances 

Let's get started!
"""

import os
from functools import partial
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn

import pyro as py
import pyro.distributions as dist
import pyro.infer
import pyro.optim
import pyro.poutine as poutine

import matplotlib.pyplot as plt 

# Getting dataset for this problem
DATA_URL = "https://d2fefpcigoriu7.cloudfront.net/datasets/rugged_data.csv"
data = pd.read_csv(DATA_URL, encoding="ISO-8859-1")
df = data[["cont_africa", "rugged", "rgdppc_2000"]]
df = df[np.isfinite(df.rgdppc_2000)]
df["rgdppc_2000"] = np.log(df["rgdppc_2000"])

# Visualize the data
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
african_nations = data[data["cont_africa"] == 1]
non_african_nations = data[data["cont_africa"] == 0]
sns.scatterplot(non_african_nations["rugged"],
            np.log(non_african_nations["rgdppc_2000"]),
            ax=ax[0])
ax[0].set(xlabel="Terrain Ruggedness Index",
          ylabel="log GDP (2000)",
          title="Non African Nations")
sns.scatterplot(african_nations["rugged"],
            np.log(african_nations["rgdppc_2000"]),
            ax=ax[1])
ax[1].set(xlabel="Terrain Ruggedness Index",
          ylabel="log GDP (2000)",
          title="African Nations")

plt.show()

# Defining a linear model

def model(cont_africa, rugged):
    a = py.sample("a", dist.Normal(8.,1000.))
    b_a = py.sample("bA", dist.Normal(0.,1.))
    b_r = py.sample("bR", dist.Normal(0.,1.))