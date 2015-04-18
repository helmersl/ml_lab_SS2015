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
    Z = numpy.dot(U[:,0:m]).T, X_mean)
    return Z, U, D

def gammaidx(X, k):
    ''' your header here!
    '''
    def distmat(X):
        


def lle(X, m, n_rule, param, tol=1e-2):
    ''' your header here!
    '''
    print 'Step 1: Finding the nearest neighbours by rule ' + n_rule
    
    print 'Step 2: local reconstruction weights'
    
    print 'Step 3: compute embedding'
