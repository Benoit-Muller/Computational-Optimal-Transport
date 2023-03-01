''' Implementation of the Hungarian algorithm.
Follows the notations and pseudocode of "Assignment Problems" by R. Burkard, M. Dell’Amico, and S. Martello.
'''
import numpy as np
import warnings

def preprocess(C):
    ''' Compute a feasible dual solution (U and V) and partial primal solution (row,x) for a cost C.
        (return vectors as 1-dim arrays)

    '''
    n,n=np.shape(C)
    U=np.min(C,axis=1)
    V=np.min(C-U[:,np.newaxis],axis=0)
    assert np.all(U[:,np.newaxis] + V <= C), "dual variables not feasible"
    row= np.full(n,None)
    x = np.full((n,n),False) # x not used for hungarian algo, take it off later
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
    SU = np.zeros(n).astype(bool) # scanned in U (also labelised)
    LV = np.zeros(n).astype(bool) # labelised in V 
    SV = np.zeros(n).astype(bool) # scanned in V
    pred = np.full(n,None) # predecessor labelisation of the tree
    fail = False # fail when cannot find a augmenting tree
    sink = None # the final node of an augmenting tree
    i = k # current vertex in U
    while (not fail) and sink is None:
        # each iteration try to go to V and come back 
        print(i)
        SU[i] = True # i is scanned:
        for j in range(n): # scanning of i
            #print((not SV[j]), np.isclose(U[i]+V[j],C[i,j],rtol=1e-3))
            if (not LV[j]) and np.isclose(U[i]+V[j],C[i,j],rtol=1e-8):
                print(" ",j)
                pred[j]=i
                LV[j] = True

        # look for a vertex j in V labelised but unscanned: 
        j=-1
        while j<n and ((not LV[j]) or SV[j]):
            j=j+1
        if j==n: # j not found
           fail = True
        else: # j found
            SV[j] = True 
            if row[j] is None: # j unmached, found a augmenting path
                sink = j
            else: # j is matched, continue the tree there
                i = row[j]
    return sink,pred,SU,LV

def hungarian(C):
    ''' O(n^4) Hungarian algorithm '''
    n,U,V,row,x = preprocess(C) # attention, x not used anymore
    phi = np.empty(n) # i=row[j] iff phi[i]==j
    AU = np.full(n,False) # assigned vertex in U
    for i in range(n):
        if not (row[i] is None):
            phi[row[i]] = i
            AU[i]=True
    while not np.all(AU): # while the assigment is partial(not a matching)
        k = np.flatnonzero(1-AU)[0] # take the first available root for alternate()
        print("k=",k)
        while AU[k] is False:
            sink,pred,SU,LV = alternate(C,U,V,row,k) # grow alternating tree
            if not(sink is None): # tree is augmenting, increase primal solution:
                AU[k]=True 
                j = sink
                condition = True
                while condition: # while we haven't reach the root
                    i = pred[j]
                    row[j] = i
                    phi[i],j = j,phi[i]
                    condition = i==k
            else: # tree is not augmenting, update dual solution:
                delta = np.min((C-U[:,np.newaxis]-V)[SU,1-LV])
                U = U + delta*SU
                V = V - delta*LV
    assert not np.any(np.sort(row) - np.arange(n)), "primal variables not feasible"
    assert np.all(U[:,np.newaxis] + V <= C), "dual variables not feasible"
    assert not np.any((1-np.isclose(U[:,np.newaxis]+V,C)) * x), "complementary stackness not satisfied"
    return row,phi,U,V
