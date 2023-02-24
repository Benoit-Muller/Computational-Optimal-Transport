# The Hungarian algorithm
import numpy as np
import warnings

'''
to define:
matrix of costs C of size n*n
'''

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
