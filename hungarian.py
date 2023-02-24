# The Hungarian algorithm
import numpy as np
import warnings

'''
to define:
matrix of costs C of size n*n
'''

def preprocess(C):
    ''' Compute a feasible dual solution (U and V) and partial primal solution (row,x) for a cost C.
        return vectors as 1-dim arrays

    '''
    n,n=np.shape(C)
    U=np.min(C,axis=1)
    V=np.min(C-U[:,np.newaxis],axis=0)
    row= np.zeros(n)
    x=np.zeros((n,n)).astype(bool)
    for i in range(n):
        for j in range(n):
            if row[j]==0 and np.isclose(U[i]+V[j],C[i,j]):
                x[i,j] = True
                row[j] = i
                break
    return n,U,V,row,x


def alternate(C,U,V,row,k):
    ''' Find an alternating tree rooted at an unassigned vertex k ∈ U
    returns 
            sink: the final leaf
            pred: array of predescessors
    First version, not tested. '''
    n,n = np.shape(C)
    SU = np.zeros(n).astype(bool)
    LV = np.zeros(n).astype(bool)
    SV = np.zeros(n).astype(bool)
    pred = -np.ones(n)
    fail = False
    sink = -1
    i = k
    while not fail and sink==-1:
        SU[i] = True
        for j in range(n):
            if LV[j] and (not SV[j]) and np.isclose(U[i]+V[j],C[i,j]):
                pred[j]=i
                LV[j] = True
        j=0
        while ((not LV[j]) or SV[j]) and j<n:
            j=j+1
        if j==n:
           fail = True
        else:
            SV[j] = True
            if row[j] == -1:
                sink = j
            else:
                i = row[j]
    return sink
