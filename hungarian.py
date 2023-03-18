''' Implementation of the Hungarian algorithm.
Follows the notations and pseudocode of "Assignment Problems" by R. Burkard, M. Dell’Amico, and S. Martello.
'''
import numpy as np
#import warnings


def preprocess(C,tol=1e-5):
    ''' Compute a feasible dual solution (U and V) and partial primal solution (row,x) for a cost C.
        (return vectors as 1-dim arrays)
    '''
    n,n=np.shape(C)
    U=np.min(C,axis=1)
    V=np.min(C-U[:,np.newaxis],axis=0)
    assert np.all(U[:,np.newaxis] + V <= C + tol), "dual variables not feasible, with transgression " + str(np.min(C-U[:,np.newaxis] - V))
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
        #print("   i=",i)
        SU[i] = True # i is scanned:
        for j in range(n): # scanning of i
            #print((not SV[j]), np.isclose(U[i]+V[j],C[i,j],rtol=1e-3))
            if (not LV[j]) and np.isclose(U[i]+V[j],C[i,j],rtol=1e-8):
                #print("    label j=",j)
                pred[j]=i
                LV[j] = True

        # look for a vertex j in V labelised but unscanned: 
        j=0
        while j<n and ((not LV[j]) or SV[j]):
            j=j+1
        #print("   selected j=",j)
        if j==n: # j not found
           #print("   fail")
           fail = True
        else: # j found
            SV[j] = True 
            if row[j] is None: # j unmached, found a augmenting path
                #print("   sink=",j)
                sink = j
            else: # j is matched, continue the tree there
                i = row[j]
    return sink,pred,SU,LV

def hungarian(C,tol=1e-5,disp=True):
    ''' O(n^4) Hungarian algorithm '''
    n,U,V,row,x = preprocess(C,tol) # attention, x not used anymore
    phi = np.empty(n, np.int8) # i=row[j] iff phi[i]==j
    AU = np.full(n,False) # assigned vertex in U
    for j in range(n): # initialise phi and AU from row 
        if not (row[j] is None):
            phi[row[j]] = j
            AU[row[j]]=True # were some error here
    while not np.all(AU): # while the assigment is partial(not a matching)
        k = np.flatnonzero(1-AU)[0] # take the first available root for alternate()
        #print("row=",row)
        #print("AU=",AU)
        #print("k=",k)
        while AU[k]==False:
            #print("  alternate with k=",k)
            sink,pred,SU,LV = alternate(C,U,V,row,k) # grow alternating tree
            #print(" sink=",sink)
            #print(" pred=",pred)
            if not(sink is None): # tree is augmenting, increase primal solution:
                #print(" primal")
                #print(" pred=",pred)
                AU[k]=True 
                j = sink
                condition = True
                while condition: # while we haven't reach the root
                    i = pred[j]
                    #print("  i,j=",i,j)
                    row[j] = i
                    phi[i],j = j,phi[i]
                    condition = not(i==k)
            else: # tree is not augmenting, update dual solution:
                #print(" dual",U,V)
                delta = np.min((C-U[:,np.newaxis]-V)[SU[:,np.newaxis]*(LV==False)])
                #print("delta=",delta)
                U = U + delta*SU
                V = V - delta*LV
    #print("row=",row,type(row))
    x=np.full((n,n),False) # attention x is not updated until now:
    for j in range(n):
        x[row[j],j]=True
    #print("x=",x)
    assert not np.any(np.sort(row) - np.arange(n)), "primal variables not feasible"
    assert np.all(U[:,np.newaxis] + V <= C + tol), "dual variables not feasible, with transgression " + str(np.min(C-U[:,np.newaxis] - V))
    assert not np.any((1-np.isclose(U[:,np.newaxis]+V,C)) * x), "complementary stackness not satisfied"
    if disp == True:
        print("hungarian succed (feasibility and complementary slackness holds)")
    W = np.sum(x*C)
    return row,x,phi,U,V,W

def augment(C,U,V,row,k):
    ''' Find an alternating tree rooted at an unassigned vertex k ∈ U
    returns 
            sink: the final leaf
            pred: array of predescessors
    First version, not tested. '''
    tol=1e-5
    n,n = np.shape(C)
    pi = np.full(n,np.inf)# min of column
    SU = np.zeros(n).astype(bool) # scanned in U (also labelised)
    LV = np.zeros(n).astype(bool) # labelised in V 
    SV = np.zeros(n).astype(bool) # scanned in V
    pred = np.full(n,None) # predecessor labelisation of the tree
    sink = None # the final node of an augmenting tree
    i = k # current vertex in U
    while sink is None:
        # each iteration augment the tree: go to V and come back 
        #print(" i=",i)
        SU[i] = True # i is scanned:
        for j in range(n): # scanning of i
            #print((not SV[j]), np.isclose(U[i]+V[j],C[i,j],rtol=1e-3))
            if (not LV[j]) and C[i,j]-U[i]-V[j] < pi[j]+tol:
                #print("  i scan j=",j)
                pred[j]=i
                pi[j]= C[i,j]-U[i]-V[j]
                if pi[j]<tol:
                    LV[j] = True

        if not np.any(LV&(~SV)): # j not found, dual update
           #print("  dual update")
           delta = np.min(pi[~LV])
           #print("  delta=",delta)
           U[SU] = U[SU] + delta
           V[LV] = V[LV] - delta
           pi[~LV] = pi[~LV] - delta
           LV[pi<tol] = True
           #print(" ", np.arange(4)[pi<tol],"added")

        # augment tree:
        j = np.flatnonzero(LV&(~SV))[0]
        #print("  j=",j)
        SV[j] = True
        if row[j] is None: # j unmached, found a augmenting path
            #print("   sink=",j)
            sink = j
        else: # j is matched, continue the tree there
            i = row[j]

    return sink,pred,U,V

def hungarian3(C,tol=1e-5,disp=True):
    ''' O(n^3) Hungarian algorithm '''
    n,U,V,row,x = preprocess(C,tol) # attention, x not used anymore
    phi = np.empty(n, np.int8) # i=row[j] iff phi[i]==j
    AU = np.full(n,False) # assigned vertex in U
    for j in range(n): # initialise phi and AU from row 
        if not (row[j] is None):
            phi[row[j]] = j
            AU[row[j]]=True # were some error here
    #print("AU=",AU)
    while not np.all(AU):
        k = np.flatnonzero(~AU)[0]
        #print("k=",k,"from AU=",AU)
        sink,pred,U,V = augment(C,U,V,row,k)
        #print("sink=",sink)
        AU[k] = True
        j = sink
        condition = True
        while condition:
            i = pred[j]
            row[j] = i
            phi[i],j = j,phi[i]
            condition = i!=k

    x=np.full((n,n),False) # attention x is not updated until now:
    for j in range(n):
        x[row[j],j]=True
    #print("x=",x)
    #print(row)
    assert not np.any(np.sort(row) - np.arange(n)), "primal variables not feasible"
    assert np.all(U[:,np.newaxis] + V <= C + tol), "dual variables not feasible, with transgression " + str(np.min(C-U[:,np.newaxis] - V))
    assert not np.any((1-np.isclose(U[:,np.newaxis]+V,C,atol=tol)) * x), "complementary stackness not satisfied"
    if disp == True:
        print("hungarian3 succed (feasibility and complementary slackness holds)")
    W = np.sum(x*C)
    return row,x,phi,U,V,W
