# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 10:47:47 2020

@author: spinosa

"homework 1"

https://stackoverflow.com/questions/28342968/how-to-plot-a-2d-gaussian-with-different-sigma

https://www.youtube.com/watch?v=eho8xH3E6mE 

https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.multivariate_normal.html
https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.random.multivariate_normal.html
"""


from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt 
import math 



x, y = np.mgrid[-4:3:.01, -3:4:.01]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x; pos[:, :, 1] = y


mean=[0.0, 0.5]
cov= [[2.0,1.0],[1.0,2.0]]
rv = multivariate_normal(mean, cov)
sample = np.random.multivariate_normal(mean, cov, 100)
#print(sample)

plt.scatter(sample[:,0],sample[:,1])

average = np.mean(sample, axis=0)
#
plt.plot(average[0], average[1], marker='o', markersize = 5, color ='red')
#
cs = plt.contour(x,y, rv.pdf(pos), color = 'green')
plt.clabel(cs, inline=1, fontsize=8)
#
v=np.array([[1/math.sqrt(2),-1/math.sqrt(2)],[math.sqrt(3)/math.sqrt(2),math.sqrt(3)/math.sqrt(2)]])



#origin = [0],[0.5]
av = np.average(sample)
origin = [0], [av]

plt.quiver(*origin, v[:,0], v[:,1], color = ['r', 'b'], scale=5)
plt.show()

#gaussian = 1/(cov * np.sqrt(2 * np.pi)) * np.exp( - (bins - mean)**2 / (2 * cov**2) )
        

'''

NEW 

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

x, y = np.random.multivariate_normal(mean, cov, 100).T

plt.figure()
plt.plot(x, y, 'o')
plt.axis('equal')
plt.show()

plt.plot(mean[0], mean[1], marker='x', markersize = 5, color ='red')

origin = [0], [0.5]
v=np.array([[1/math.sqrt(2),-1/math.sqrt(2)],[math.sqrt(3)/math.sqrt(2),math.sqrt(3)/math.sqrt(2)]])
plt.quiver(*origin, v[:,0], v[:,1], color = ['r', 'b'], scale=5)
plt.show()

#xx, yy = np.mgrid[-2:1:.01, -2:1:.01]
xx, yy = np.mgrid[np.min(x):np.max(x):.01, np.min(y):np.max(y):.01]
pos = np.empty(xx.shape + (2,))
pos[:, :, 0] = xx; pos[:, :, 1] = yy

mu = np.array([0., 0.5])
sigma = np.array([ [2.0, 1.0] , [1.0, 2.0] ])

Z = multivariate_gaussian(pos, mu, sigma)

cp = plt.contour(xx, yy, Z)
plt.clabel(cp, inline=True, 
          fontsize=10)
plt.show()

plt.figure()
pos[:, :, 0] =xx; pos[:, :, 1] = 2
plt.plot(xx, rv.pdf(pos), 'blue')
plt.axvline(x=0.75, color='r--')

'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Our 2-dimensional distribution will be over variables X and Y
N = 40
X = np.linspace(-5, 5, N)
Y = np.linspace(-5, 5, N)
X, Y = np.meshgrid(X, Y)

# Mean vector and covariance matrix
mu = np.array([0., 0.5])
Sigma = np.array([ [2.0, 1.0] , [1.0, 2.0] ])

# Pack X and Y into a single 3-dimensional array
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

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

# The distribution on the variables X, Y packed into pos.
Z = multivariate_gaussian(pos, mu, Sigma)

# plot using subplots
fig = plt.figure()
ax1 = fig.add_subplot(2,1,1,projection='3d')

ax1.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
                cmap=cm.viridis)
ax1.view_init()
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_zticks([])
ax1.set_xlabel(r'$x_1$')
ax1.set_ylabel(r'$x_2$')

ax2 = fig.add_subplot(2,1,2,projection='3d')
ax2.contourf(X, Y, Z, zdir='z', offset=0, cmap=cm.viridis)
ax2.view_init()

ax2.grid(False)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_zticks([])
ax2.set_xlabel(r'$x_1$')
ax2.set_ylabel(r'$x_2$')

plt.show()


'''

# Our 2-dimensional distribution will be over variables X and Y
N = 60
X = np.linspace(-3, 3, N)
Y = np.linspace(-3, 4, N)
X, Y = np.meshgrid(X, Y)

# Mean vector and covariance matrix



mean=np.array([0.0, 0.5])
cov= np.array([[2.0,1.0],[1.0,2.0]])

# Pack X and Y into a single 3-dimensional array
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.

    """
    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N

# The distribution on the variables X, Y packed into pos.
Z = multivariate_gaussian(pos, mean,cov)

# Create a surface plot and projected filled contour plot under it.
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
                cmap=cm.viridis)

cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=cm.viridis)

# Adjust the limits, ticks and view angle
ax.set_zlim(-0.15,0.2)
ax.set_zticks(np.linspace(0,0.2,5))
ax.view_init(27, -21)
ax.plot(np.ones(N)*X[0,19], Y[:,19], Z[:,19], lw=5, c='r', zorder=1000)
plt.show()

'''
