"""
Intro to Probabilistic Programming: Models and Inference in Pyro

by Sourabh Kulkarni (https://www.github.com/SourabhKul)

Following instructions from MLTrain@UAI 2018 Pyro Workshop

Some basics before we get started:

Why Probabilistic Programming? Powerful generative models:
- Character Generation 
- Scene Representation
- Physics Engines: Simulating real worlds
- Automated Statistician: Perform statistical analysis automatically

Principles of Probabilistic Inference:
- Sum Rule: P(x) = sum_y(P(x,y))
- Product Rule = P(x,y) = P(x)p(y|x)
- Bayes Rule = P(theta|data) = P(data|theta)P(theta)/P(data)

Some Terminology:
- P(theta|data) is the posterior of theta given data
- P(data|theta) is the likelihood of theta
- P(theta) is the prior over theta

Probabilistic Programming Languages:
- Uniform, Universal specification of processes with high-level abstractions

Recipe:
- High-level Programming Language (python)
- Primitives of random objects as probability distributions (random/stochastic functions)
- Sampling Statements (draw samples from random functions)
- Perform Inference (computing posterior)

Let's get started!
"""
import numpy as np
import torch
import pyro as py
import pyro.distributions as dist
import pyro.infer
import pyro.optim

import matplotlib.pyplot as plt 

def weather(p_cloudy):
    """
    This is an example of a stochastic function. It predicts the weather given the probability of it being cloudy
    input:
        - p_cloudy(float): probability that it is cloudy outside
    returns:
        - is_cloudy(boolean): is it cloudy?
        - tempreature(float): a sample from the distribution of weather given is_cloudy
    """
    is_cloudy = dist.Bernoulli(p_cloudy).sample()
    
    if is_cloudy:
        loc, scale = 55.0, 10.0
    else:
        loc, scale = 75.0, 15.0
    
    temperature = py.sample("temp", dist.Normal(loc,scale))

    return is_cloudy.item(), temperature.item()

def ice_cream_sales():
    """
    Another stochastic function that estimates ice cream sales based on weather
    inputs:
        - none
    returns:
        - sample from the distibution of expected sales
    """
    is_cloudy, temperature = weather(0.3)

    if not is_cloudy and temperature >= 80:
        expected_sales = 200
    else:
        expected_sales = 50
    
    return py.sample('sales', dist.Normal(expected_sales,10))

# print (ice_cream_sales())

def scale(guess):
    """
    In this stochastic function, we have an initial guess of the weight and we assume
    the true weight is a gaussian with mean value of the guess and sigma of 1. We then
    model the noisy scale with a gaussian with mean as the true weight and variance of
    0.85 and return a sample from that distribution
    inputs:
        - guess: initial guess of the weight
    returns:
        - measurement: the weight as measured by the noisy scale 
    """
    weight = py.sample("weight", dist.Normal(guess,1))
    return py.sample("measurement", dist.Normal(weight,0.75))

# We can condition the scale function by the observed measurement as follows
observed_measurement = 9.5
conditioned_scale = py.condition(scale, data={"measurement":observed_measurement})

# We can now infer over the posterior of the function given the observed measurement
posterior = py.infer.Importance(conditioned_scale, num_samples=1000)

# We can also infer the marginal distribution over the true weight, given our original guess and the observed measurement
guess = 8
marginal = py.infer.EmpiricalMarginal(posterior.run(guess), sites="weight")

plt.hist(np.array(marginal.sample((1000,))))
plt.title("p(weight|measurement="+str(observed_measurement)+"guess="+str(guess))
plt.show()

