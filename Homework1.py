# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 10:47:47 2020

@author: spinosa
"""

from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt 
import math 

#x = np.linspace(0, 5, 10, endpoint=False)
#y = multivariate_normal.pdf(x, mean=2.5, cov=0.5); 
#rv = multivariate_normal(mean=[0,0.5], cov=[[2.0,1.0],[1.0,2]])
#>>> plt.plot(x, y)

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
origin = [0],[0.5]

plt.quiver(*origin, v[:,0], v[:,1], color = ['r', 'b'], scale=5)
plt.show()

#gaussian = 1/(cov * np.sqrt(2 * np.pi)) * np.exp( - (bins - mean)**2 / (2 * cov**2) )
        



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
