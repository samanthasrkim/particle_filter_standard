# Snyder_main.py
'''
Code to run the simulations from Snyder et al. 2008
Obstacle to High-Dimensional Particle Filtering

This code gives the basic particle filter algorithm and can be copied and modified to compute specific results

By Samantha Kim 2021
'''

# Definitions
'''
1) Data model Y = HX + eps
Y is a (d x 1) vector
H is a (d x q) matrix
X (truth) is a (q x 1) vector
eps (not sure) (d x 1) vector
(Here d=q because Nx=Ny)

2) Proposal (prior) distribution is Gaussian
p(X) = N(0, sigma_x)
sigma_x is the identity matrix Id (d x d).
x_p is a (d x Ne) array

3) Error distribution is Gaussian
p(eps) = N(0, sigma_eps)
sigma_eps is the identity matrix, Id (d x d).

4) The likelihood is Gaussian
p(Y|X) ~ N(HX, Id)

5) The weights are calculated given the eq.(3) in Snyder et al. 2008 and the eq.(4) in Bengtsson et al.2008
        exp{-0.5 ||Y - HXi||^2}
wi =----------------------------
      n
     SUM exp{-0.5 ||Y - HXj||^2}
     j=1

6) Each parameter has a prior distribution
7) Each parameter has a likelihood (still no sure hox I will compte that though)
8) 1 vector (Ne x 1) for the weights
'''



# Library
import numpy as np
from numpy import *
import math

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from random import *
import statistics as stat

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import pylab as plt
import numpy.matlib
import scipy.integrate as integrate

from numpy.random import randn
from numpy.linalg import inv

plot = 1

## Set the experiment 
n_run = 400
save_file = 1
w_max = np.zeros((3, n_run))

## Set the dimension
dim = np.array([10, 30, 100])

## Set the particle filter
Ne = 1000

for d in range(len(dim)):
    Ny = dim[d]

    ## Initialization
    Id = np.identity(Ny)
    # Truth
    x_truth =  np.random.multivariate_normal(np.zeros(Ny), Id)
    # Prior
    x_p = np.zeros((Ny, Ne))
    # Data model
    H = Id
    # Error distribution
    eps = np.zeros((Ny, 1))
    # Data
    y = np.zeros((Ny, 1))

    ## Set true value of the parameters
    cov_prior = np.identity(Ne)
    x_truth = np.random.multivariate_normal(np.zeros(Ny), Id)
    
    
    for simu in range(n_run):
        print(simu)
        ## Create the priors: 1 prior distribution per parameter
        for i in range(Ny):
            # each row is an ensemble of Ne values for the i:th parameter drawn from p(X) too
            x_p[i, :] =  np.random.multivariate_normal(np.zeros(Ne), cov_prior)
        
        
        ## Create the observations
        # Create a separated vector for eps to save the input errors
        eps = np.random.multivariate_normal(np.zeros(Ny), Id)
        y = x_truth + eps

        ## Filtering part 
        sigma_lik = 1
        w = np.ones(Ne, dtype=np.float64)
        dlpsi = np.zeros((Ne, Ny), dtype=np.float64)
        # Calculation of the diff obs-model (model computed with the particles)
        diff = np.zeros((Ny, Ne))
        diff_squared = np.zeros((Ny, Ne))
        p_yx = np.zeros((Ny, Ne))

        stepbystep = 1
        if stepbystep == 1:
        # Here I make it step by step to come back if there is a problem because it's easier to checked the results
            for i in range(Ne):
                diff[:, i] = y - x_p[:, i] # checked
                diff_squared[:, i] = diff[:, i]**2 # checked
            p_yx = (1/sqrt(2*pi*sigma_lik))*np.exp(-0.5*diff_squared/(sigma_lik**2)) 

            # compute the total probability for the likelihood because y is a vector p(y|xi) = p(y1|xi) x ... x p(y_Ny|xi)
            p_yx_tot = np.ones(Ne)
            for i in range(Ne): # can be shorten after with only one loop
                for j in range(Ny):
                    p_yx_tot[i] = p_yx_tot[i] * p_yx[j, i]

            # Calculation of the weights
            for i in range(Ne):
                w[i] = p_yx_tot[i] / np.sum(p_yx_tot)


        ## Rapid check with the histograms of the w_max
        w_max[d, simu] = np.max(w)
    
savetxt('output/weights.txt', w_max)

if plot == 1:
    # Plot the histogram for the first dimension (Nx=10)
    fig = plt.figure(figsize=(15,10))
    n, bins, patches = plt.hist(w_max[0, :], 50, facecolor='green', alpha=50)
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(0, 300)
    plt.xlabel('max $\mathrm{w_i}$')
    plt.ylabel('occurrences')
    plt.savefig('outputfig/histogramNx10_Fig1b.png')
    
    
    fig = plt.figure(figsize=(15,10))
    n, bins, patches = plt.hist(w_max[1, :], 50, facecolor='green', alpha=50)
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(0, 300)
    plt.xlabel('max $\mathrm{w_i}$')
    plt.ylabel('occurrences')
    plt.savefig('outputfig/histogramNx30_Fig1b.png')


    fig = plt.figure(figsize=(15,10))
    n, bins, patches = plt.hist(w_max[2, :], 50, facecolor='green', alpha=50)
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(0, 300)
    plt.xlabel('max $\mathrm{w_i}$')
    plt.ylabel('occurrences')
    plt.savefig('outputfig/histogramNx100_Fig1b.png')







