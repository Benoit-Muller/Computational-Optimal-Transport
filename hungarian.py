''' Implementation of the Hungarian algorithm.
Follows the notations and pseudocode of "Assignment Problems" by R. Burkard, M. Dell’Amico, and S. Martello.
'''
import numpy as np
import warnings

def preprocess(C):
    ''' Compute a feasible dual solution (U and V) and partial primal solution (row,x) for a cost C.
        return vectors as 1-dim arrays

    '''
    n,n=np.shape(C)
    U=np.min(C,axis=1)
    V=np.min(C-U[:,np.newaxis],axis=0)
    assert np.all(U[:,np.newaxis] + V <= C), "dual variables not feasible"
    row= np.full(n,None)
    x = np.full((n,n),False)
    for i in range(n):
        for j in range(n):
            if row[j] is None and np.isclose(U[i]+V[j],C[i,j]):
                x[i,j] = True
                row[j] = i
                break
    assert not np.any((1-np.isclose(U[:,np.newaxis]+V,C)) * x), "complementary stackness not satisfied"
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
    pred = np.full(n,None)
    fail = False
    sink = None
    i = k
    while (not fail) and sink is None:
        print(i)
        SU[i] = True
        for j in range(n):
            #print((not SV[j]), np.isclose(U[i]+V[j],C[i,j],rtol=1e-3))
            if (not LV[j]) and np.isclose(U[i]+V[j],C[i,j],rtol=1e-8):
                print(" ",j)
                pred[j]=i
                LV[j] = True
        j=-1
        while ((not LV[j]) or SV[j]) and j<n-1:
            j=j+1
        if j==n:
           fail = True
        else:
            SV[j] = True
            if row[j] is None:
                sink = j
            else:
                i = row[j]
    return sink,pred,SU,LV

def hungarian(C):
    ''' O(n^4) Hungarian algorithm '''
    n,U,V,row,x = preprocess(C) # attention, x not used anymore
    phi = np.empty(n)
    for i in range(n):
        if not (row[i] is None):
            phi[row[i]] = i
    AU = np.full(n,False)
    while not np.all(AU):
        k = np.nonzero(AU)[0]
        while AU[k] is False:
            sink,pred = alternate(C,U,V,row,k)
            if not(sink is None): #increase the primal solution:
                AU[k]=True
                j = sink
                condition = True
                while condition:
                    i = pred[j]
                    row[j] = i
                    phi[i],j = j,phi[i]
                    condition = i==k
            else: # update the dual solution:
                delta = np.min((C-U[:,np.newaxis]-V)[SU,1-LV])
                U = U + delta*SU
                V = V - delta*LV
    assert not np.any(np.sort(row) - np.arange(n)), "primal variables not feasible"
    assert np.all(U[:,np.newaxis] + V <= C), "dual variables not feasible"
    assert not np.any((1-np.isclose(U[:,np.newaxis]+V,C)) * x), "complementary stackness not satisfied"
    return row,phi,U,V
