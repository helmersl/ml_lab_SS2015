""" sheet1_implementation.py

PUT YOUR NAME HERE:
Lea Helmers
<FIRST_NAME><LAST_NAME>


Write the functions
- pca
- gammaidx
- lle
Write your implementations in the given functions stubs!


(c) Daniel Bartz, TU Berlin, 2013
"""
import numpy as np
import scipy.linalg as la
from scipy.spatial import distance

def pca(X, m):
    ''' pca - compute principal components

        usage:
           Z, U, D = pca(X, m)

        input:
           X : (d,n)-array containing one data point per column
           m : number of principal components

        output:
           Z : (m,n)-array, data projected to the first m components 
           U : 
           D : 
        description:

    '''
    #Calculate principal components
    X_mean = np.mean(X, axis=1)[:,np.newaxis]
    C = X-X_mean
    S = np.cov(C)
    l,U = np.linalg.eigh(S)
    idx = np.argsort(l)[::-1]
    U = U[:,idx]
    D = l[idx]

    #Project the data from d- to m-dimensional space
    Z = np.dot((U[:,0:m]).T, C)
    return Z, U, D

def gammaidx(X, k):
    ''' your header here!
    '''
    distmat = distance.squareform(distance.pdist(X.T))
    distmat.sort(axis=1)
    
    return np.mean(distmat[:,1:k+1], axis=1)[np.newaxis]

def lle(X, m, n_rule, param, tol=1e-2):
    ''' your header here!
    '''
    print 'Step 1: Finding the nearest neighbours by rule ' + n_rule
    dims, n = X.shape
    W = np.zeros([n,n])
    distmat = distance.squareform(distance.pdist(X.T))
    neighbour_indices = distmat.argsort(axis=1)
    if n_rule == 'knn':
        k = param
        one_vector = np.ones(k)
        #Construct weight matrix
        print 'Step 2: local reconstruction weights'
        for i in range(n):
            nearest_neighbours = X[:,neighbour_indices[i,1:k+1]]
            nearest_neighbours_centered = nearest_neighbours - X[:,i].reshape(dims,1)
            C = np.cov(nearest_neighbours_centered.T)
            C_reg = C + tol*np.eye(k)
            w = np.linalg.solve(C_reg, one_vector)
            w = w/w.sum()
            W[i,neighbour_indices[i,1:k+1]] = w
    if n_rule == 'eps-ball':
        print 'Step 2: local reconstruction weights'
        print len(distmat[distmat < param])
        if np.array_equal(distmat<param, np.eye(distmat.shape[0])):
            raise ValueError('eps is too small, no nearest neighbours can be found')
        for i in range(n):
            mask = distmat[i,:]<param
            mask[i] = False
            nearest_neighbours_idx = np.arange(0,n)[mask]
            k = len(nearest_neighbours_idx)
            if k == 0: continue
            one_vector = np.ones(k)
            nearest_neighbours = X[:,nearest_neighbours_idx]
            nearest_neighbours_centered = nearest_neighbours - X[:,i].reshape(dims,1)
            C = np.cov(nearest_neighbours_centered.T)
            C_reg = C + tol*np.eye(k)
            w = np.linalg.solve(C_reg, one_vector)
            w = w/w.sum()
            W[i,nearest_neighbours_idx] = w
    
    print 'Step 3: compute embedding'
    M = ((np.eye(n)-W).T).dot(np.eye(n)-W)
    e,E = np.linalg.eigh(M) 
    E = E[:,e.argsort()]
    return (E[:,1:m+1]).T
