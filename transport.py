''' Functions related to known optimal transport maps'''
import numpy as np
from scipy.linalg import sqrtm,inv,norm
from scipy.stats import qmc
from warnings import warn
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def random_gaussian_parameters(d,rng):
    """Compute a random pair of mean and covariance, in dimension d.
    (could use scipy.stats.random_correlation)
    """
    mean = rng.normal(size=d)
    A = rng.normal(size=(d,d))
    ii=np.arange(d)
    jj=np.arange(d)[:,np.newaxis]
    A[ii-jj<0] = 0
    A[ii-jj==0] = np.abs(A[ii-jj==0])
    cov = A.T @ A
    return mean, cov

def gaussian_transport(mean1, cov1, mean2, cov2):
    ''' Compute the the Wasserstein distance bewteen two gaussian measure,
        and its associated optimal transport map. '''
    tol = 1e-10
    if np.abs(cov1-cov1.T).max() > tol or np.abs(cov2-cov2.T).max() > tol:
        raise Exception("Covariance matrix must be symmetric")
    A1 = sqrtm(cov1)
    A2 = sqrtm(cov1)
    B = sqrtm(A1 @ cov2 @ A1)
    A = inv(A1) @ B @ inv(A1)
    transport = lambda x: A@(x-mean1) + mean2
    W_cost = np.linalg.norm(mean1 - mean2)**2 + np.trace(cov1 + cov2 - 2*B)
    return transport, W_cost

def gaussian_discreatization(mean1, cov1, mean2, cov2, n, rng):
    x = rng.multivariate_normal(mean1, cov1, size=n) 
    y = rng.multivariate_normal(mean2, cov2, size=n)
    #dist = qmc.MultivariateNormalQMC(mean1, cov1, seed=rng)
    #x = dist.random(n)
    #dist = qmc.MultivariateNormalQMC(mean2, cov2, seed=rng)
    #y = dist.random(n)
    C = np.sum((x[:,np.newaxis,:] - y[np.newaxis,:,:])**2, axis=2)/n
    return x, y, C

def torus_dist2(x,y):
    dist = 0
    for i in range(len(x)):
        dist += np.min([(x[i]-y[i])**2, (1-np.abs(x[i]-y[i]))**2])
    return dist

def periodic_gaussian_discreatization(mean1, cov1, mean2, cov2, n, rng):
    x = rng.multivariate_normal(mean1, cov1, size=n) 
    y = rng.multivariate_normal(mean2, cov2, size=n)
    x = x % 1
    y = y % 1
    #dist = qmc.MultivariateNormalQMC(mean1, cov1, seed=rng)
    #x = dist.random(n)
    #dist = qmc.MultivariateNormalQMC(mean2, cov2, seed=rng)
    #y = dist.random(n)
    C = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            C[i,j] = torus_dist2(x[i],y[j])
    return x, y, C