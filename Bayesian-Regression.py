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

import numpy as np
import torch
import pyro as py
import pyro.distributions as dist
import pyro.infer
import pyro.optim

import matplotlib.pyplot as plt 

