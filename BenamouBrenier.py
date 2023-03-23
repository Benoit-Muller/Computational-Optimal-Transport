''' Implementation of the Benamou_Brenier method based on fluid dynamics.
'''
import numpy as np

class TransportProblem:
    def __init__(self,mesh,mu,nu):
        self.mesh = mesh
        self.mu = mu
        self.nu = nu

    def k(rho,m):
        tol = 1e-7
        norm2_m = np.sum(m**2)
        if rho >= tol:
            return 0.5 * norm2_m / rho
        if np.abs(rho) < tol:
            if norm2_m < tol:
                return 0
            else:
                return np.inf
        if rho <= -tol:
            return np.inf



def poisson_step():
    return
def projection_step():
    return
def dual_step():