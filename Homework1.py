# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 10:47:47 2020

@author: spinosa anna

"homework 1"

https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.multivariate_normal.html
https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.random.multivariate_normal.html
"""


import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt 
import math 

def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos."""

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N
    
%matplotlib auto

mean=[0.0, 0.5]
cov= [[2.0,1.0],[1.0,2.0]]
rv = multivariate_normal(mean, cov)

x, y = np.random.multivariate_normal(mean, cov, 100).T

origin = [0], [0.5]
v=np.array([[1/math.sqrt(2),-1/math.sqrt(2)],[math.sqrt(3)/math.sqrt(2),math.sqrt(3)/math.sqrt(2)]])

#xx, yy = np.mgrid[-2:1:.01, -2:1:.01]
xx, yy = np.mgrid[np.min(x):np.max(x):.01, np.min(y):np.max(y):.01]
pos = np.empty(xx.shape + (2,))
pos[:, :, 0] = xx; pos[:, :, 1] = yy

mu = np.array([0., 0.5])
sigma = np.array([ [2.0, 1.0] , [1.0, 2.0] ])

Z = multivariate_gaussian(pos, mu, sigma)




plt.figure()
plt.plot(x, y, 'o')
plt.axis('equal')
plt.plot(mean[0], mean[1], marker='x', markersize = 10, color ='red', label = 'mean')
plt.quiver(*origin, v[:,0], v[:,1], color = ['r', 'b'], scale=5)
plt.legend()
cp = plt.contour(xx, yy, Z)
plt.clabel(cp, inline=True, 
          fontsize=10)
plt.show()



plt.figure()
pos[:, :, 0] =xx; pos[:, :, 1] = 2
plt.plot(xx, rv.pdf(pos), 'blue')
plt.axvline(x=0.75, color='r', ls= '--', label='maximum for x=0.75')
plt.axvline(x=1.99, color='g', ls= '--', label = 'inflection point x=1.99')
plt.axvline(x=-0.47, color='g', ls= '--', label = 'inflection point x=-0.47')
plt.legend()
