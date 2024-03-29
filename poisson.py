# implementation of the 3D poisson equation with one heterogenous Neumann and two periodic bc.
import numpy as np
import scipy as sp
from scipy import sparse
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from matplotlib.widgets import Slider

def laplacian_matrix(n):
    An,Ap = derivative_matrices(n)
    In = sparse.eye(n,format="csr")
    Ip = sparse.eye(n-1,format="csr")
    D2x = sparse.kron(Ip,sparse.kron(Ip,An))
    D2y = sparse.kron(Ip,sparse.kron(Ap,In))
    D2z = sparse.kron(Ap,sparse.kron(Ip,In))
    A = D2x + D2y + D2z
    return A

def extend(g):
    n = np.shape(g)[1]
    G = np.zeros((n,n,n))
    G[0],G[-1] = g[1],g[0] # ?
    return G

def normalize_lagrange(A,b):
    n=len(b)
    one = sparse.bsr_matrix(np.ones((n,1)))
    A = sparse.vstack((A, one.T))
    one_zero = sparse.vstack((one, sparse.bsr_matrix([0])))
    A = sparse.hstack((A, one_zero))
    b = np.hstack((b, [0]))
    return A,b

def add_mean_condition(A,b):
    N = len(b)
    one = sparse.bsr_matrix(np.ones((1,N)))
    A = sparse.vstack((A, one))
    b = np.hstack((b, [0]))
    return A,b

def normalize_integral(b):
    return b - np.mean(b)

def poisson(f,g,A=None):
    n = np.shape(f)[0]
    order = "F"
    if A is None:
        A = laplacian_matrix(n)
    g_contribution = extend(g)[:,0:-1,0:-1]
    g_contribution[-1] = - g_contribution[-1]
    b = (f[:,0:-1,0:-1] + 2*n*g_contribution).flatten(order) / n**2

    b = b - np.sum(b)
    A,b = normalize_lagrange(A,b)
    u_vect = spsolve(A,b)
    u_vect = u_vect[:-1] 

    #b = b - np.sum(b)
    #A,b = add_mean_condition(A,b)
    #u_vect,res,rank,s = np.linalg.lstsq(A.toarray(),b)
    
    u = np.zeros((n,n,n))
    u[:,0:-1,0:-1] = u_vect.reshape((n,n-1,n-1),order=order)
    u[:,-1,0:-1] = u[:,0,0:-1]
    u[:,:,-1] = u[:,:,0]
    return u

def build_Asolve(n):
    A = laplacian_matrix(n)
    one = sparse.bsr_matrix(np.ones((n*(n-1)**2,1)))
    A = sparse.vstack((A, one.T))
    one_zero = sparse.vstack((one, sparse.bsr_matrix([0])))
    A = sparse.hstack((A, one_zero))
    Asolve = sparse.linalg.factorized(A)
    return Asolve


def poisson2(f,g,Asolve=None):
    n = np.shape(f)[0]
    if Asolve is None:
        Asolve = build_Asolve(n)
    order = "F"
    g_contribution = extend(g)[:,0:-1,0:-1]
    g_contribution[-1] = - g_contribution[-1]
    b = (f[:,0:-1,0:-1] + 2*n*g_contribution).flatten(order) / n**2
    b = b - np.sum(b)
    b = np.hstack((b, [0]))
    u_vect = Asolve(b)
    u_vect = u_vect[:-1] 

    #b = b - np.sum(b)
    #A,b = add_mean_condition(A,b)
    #u_vect,res,rank,s = np.linalg.lstsq(A.toarray(),b)
    
    u = np.zeros((n,n,n))
    u[:,0:-1,0:-1] = u_vect.reshape((n,n-1,n-1),order=order)
    u[:,-1,0:-1] = u[:,0,0:-1]
    u[:,:,-1] = u[:,:,0]
    return u

def derivative_matrices(n):
    Ap = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(n-1, n-1))
    Ap = Ap.toarray()
    Ap[0,-1], Ap[-1,0] = 1, 1
    Ap = sparse.csc_matrix(Ap)

    An = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(n, n))
    An = An.toarray()
    An[0,1], An[-1,-2] = 2, 2
    An = sparse.csc_matrix(An)
    return An, Ap

def divergence(field,An,Ap):
    d = np.empty_like(field[0])
    d[:,0:-1,0:-1] = np.einsum("ij,jkl->ikl",An.toarray(),field[0,:,0:-1,0:-1])
    d[:,0:-1,0:-1] = d[:,0:-1,0:-1] + np.einsum("ij,kjl->kil",Ap.toarray(),field[1,:,0:-1,0:-1])
    d[:,0:-1,0:-1] = d[:,0:-1,0:-1] + np.einsum("ij,klj->kli",Ap.toarray(),field[2,:,0:-1,0:-1])
    d[:,-1,0:-1] = d[:,0,0:-1]
    d[:,:,-1] = d[:,:,0]
    return d

def gradient(f,An,Ap):
    g = np.zeros((3,)+np.shape(f))
    g[0,:,0:-1,0:-1] = np.einsum("ij,jkl->ikl",An.toarray(),f[:,0:-1,0:-1])
    g[1,:,0:-1,0:-1] = np.einsum("ij,kjl->kil",Ap.toarray(),f[:,0:-1,0:-1])
    g[2,:,0:-1,0:-1] = np.einsum("ij,klj->kli",Ap.toarray(),f[:,0:-1,0:-1])
    g[:,:,-1,0:-1] = g[:,:,0,0:-1]
    g[:,:,:,-1] = g[:,:,:,0]
    return g

def plot_slider(u):
    " plot rho[i] with a slider for i. Use <%matplotlib>, and turn back to <%matplotlib inline> after. "
    fig, (ax1,ax2) = plt.subplots(2)
    s = Slider(ax = ax2, label = 'value', valmin = 0, valmax = np.shape(u)[0], valinit = 1)
    def update(val):
        value=int(s.val)
        ax1.cla()
        ax1.contour(u[value])
    s.on_changed(update)
    update(0)
    plt.show()
    return s