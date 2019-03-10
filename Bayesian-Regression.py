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

# Defining a linear model: the old way

def model(cont_africa, rugged):
    """
    Predicts the log GDP of a nation based on wether it is from Africa
    and on terrain ruggedness.
    inputs:
        - cont_africa: boolean for wether nation is from Africa 
        - rugged: ruggedness index of nation
    returns:
        - obs: sample of predicted log GDP of nation 
    """
    a = py.sample("a", dist.Normal(8.,1000.))
    b_a = py.sample("bA", dist.Normal(0.,1.))
    b_r = py.sample("bR", dist.Normal(0.,1.))
    b_ar = py.sample("bAR", dist.Normal(0.,1.))
    sigma = py.sample("sigma", dist.Uniform(0.,10.))

    mu = a + b_a * cont_africa + b_r * rugged + b_ar * cont_africa * rugged

    return py.sample("obs", dist.Normal(mu. sigma))


# But we need a uniform way to define models; so we follow the pytorch API to defining a model
# With that we define a generic regression model, and then instantiate one for our application

class RegressionModel(nn.Module):
    # generic linear regression model
    def __init__(self,p):
        # p = number of features
        super(RegressionModel, self).__init__()
        self.linear = nn.Linear(p,1)
        self.factor = nn.Parameter(torch.tensor(1.))
    
    def forward(self, x):
        return self.linear(x) + (self.factor * x[:,0] * x[:,1]).unsqueeze(1)

# now we instantiate a regression model with 2 features - is_africa, ruggedness
p = 2
logGDP_predictor = RegressionModel(p) 

# Now we learn this regression model in a bayesian way
# First we 'lift' the parameters as random variables using random_module()

loc = torch.zeros(1,1)
scale = torch.ones(1,1)

# Define a prior (unit normal)

prior = dist.Normal(loc, scale)

# Generate a random version of regression model, which will take samples as parameters

lifted_module = py.random_module("logGDP_predictor",nn,prior)

# Sample a model from prior

sampled_reg_model = lifted_module()

# Define a guide function

def guide(cont_africa, rugged, data):
    """
    Mean-field approximiation of the posterior of model parameters
    """
    loc_a = py.param("loc_a", torch.tensor(torch.randn(1)+guess))
    scale_a = py.param("scale_a", torch.randn(1))
    
    a = py.sample("a", dist.Normal(loc_a, scale_a))

    loc_b_a = py.param("loc_b_a", torch.tensor(torch.randn(1)+guess))
    scale_b_a = py.param("scale_b_a", torch.randn(1))
    
    b_a = py.sample("b_a", dist.Normal(loc_b_a, scale_b_a))

    loc_b_r = py.param("loc_b_r", torch.tensor(torch.randn(1)+guess))
    scale_b_r = py.param("scale_b_r", torch.randn(1))
    
    b_r = py.sample("b_r", dist.Normal(loc_b_r, scale_b_r))

    loc_b_ar = py.param("loc_b_ar", torch.tensor(torch.randn(1)+guess))
    scale_b_ar = py.param("scale_b_ar", torch.randn(1))
    
    b_ar = py.sample("b_ar", dist.Normal(loc_b_ar, scale_b_ar))

    sigma_dist = dist.Normal(0.,1.)
    sigma = pyro.sample("sigma", sigma_dist)