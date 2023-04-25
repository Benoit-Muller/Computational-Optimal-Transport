# implementation of the 3D poisson equation with one heterogenous Neumann and two periodic bc.
import numpy as np
import scipy as sp
from scipy import sparse
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

def laplacian_matrix(n):
    Ad = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(n, n))
    Ap = Ad.toarray()
    Ap[0,-1], Ap[-1,0] = 1, 1
    Ap = sparse.dia_matrix(Ap)
    An = Ad.toarray()
    An[0,1], An[-1,-2] = 2, 2
    An = sparse.dia_matrix(An)
    I = sparse.eye(n)
    D2x = sparse.kron(I,sparse.kron(I,An))
    D2y = sparse.kron(I,sparse.kron(Ap,I))
    D2z = sparse.kron(Ap,sparse.kron(I,I))
    A = D2x + D2y + D2z
    return A

def extend(g):
    n = np.shape(g)[1]
    G = np.zeros((n,n,n))
    G[[0,-1]] = g
    return G

def normalize_lagrange(A,b):
    n=len(b)
    one = sparse.bsr_matrix(np.ones((n,1)))
    A = sparse.vstack((A, one.T))
    one_zero = sparse.vstack((one, sparse.bsr_matrix([0])))
    A = sparse.hstack((A, one_zero))
    b = np.hstack((b, [0]))
    return A,b

def normalize_integral(b):
    return b - np.mean(b)

def poisson(f,g,A=None):
    n = np.shape(f)[0]
    order = "F"
    if A is None:
        A = laplacian_matrix(n)
    A = A * n**2
    b = (f + 2*n*extend(g)).flatten(order)

    b = normalize_integral(b)
    u_vect = spsolve(A,b)
    u_vect = normalize_integral(u_vect)
    #U= U[0:-1]
    #l = U[-1]
    u = u_vect.reshape((n,n,n),order=order)
    return u

def derivative_matrices(n):
    Ad = sparse.diags([-0.5, 0, 0.5], [-1, 0, 1], shape=(n, n))
    Ap = Ad.toarray()
    Ap[0,-1], Ap[-1,0] = -1, 1
    Ap = sparse.dia_matrix(Ap)
    An = Ad.toarray()
    An[0,0], An[-1,-1] = -1, 1
    An = sparse.dia_matrix(An)
    return An, Ap

def divergence(field,An,Ap):
    d = np.einsum("ij,jkl->ikl",An.toarray(),field[0])
    d = d + np.einsum("ij,kjl->kil",Ap.toarray(),field[1])
    d = d + np.einsum("ij,klj->kli",Ap.toarray(),field[2])
    return d

def gradient(f,An,Ap):
    g0 = np.einsum("ij,jkl->ikl",An.toarray(),f)
    g1 = np.einsum("ij,kjl->kil",Ap.toarray(),f)
    g2 = np.einsum("ij,klj->kli",Ap.toarray(),f)
    return np.stack((g0,g1,g2))