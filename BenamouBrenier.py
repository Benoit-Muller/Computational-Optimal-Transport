''' Implementation of the Benamou_Brenier method based on fluid dynamics.
'''
import numpy as np

class TransportProblem:
    def __init__(self,mesh,mu,nu,T):
        """
        With N = N_1*...*N_d discretization
        Inputs:
                mesh: array of shape            (d,N_1,...,N_d)
                mu,nu: arrays of shape            (N_1,...,N_d)
                T: integer
        Builds:
                times: array of shape          (T,)
                rho: array of shape            (T, N_1,...,N_d)
                m: array of shape           (d, T, N_1,...,N_d)
                M: array of shape         (1+d, T, N_1,...,N_d)
                phi: array of shape            (T, N_1,...,N_d)
                nabla_phi: array of shape (1+d, T, N_1,...,N_d)
                a: array of shape              (T, N_1,...,N_d)
                b: array of shape           (d, T, N_1,...,N_d)
                c: array of shape         (1+d, T, N_1,...,N_d)

        """
        # last dimensions always contain space and time  
        d = mesh.shape[0]
        space_grid_shape = mesh.shape[1:] # (N_1,...,N_d)
        spacetime_grid_shape = (T,) + space_grid_shape

        self.d = d
        self.mesh = mesh # mesh for the measure discretization
        self.mu = mu # source measure
        self.nu = nu # target measure

        self.T = T
        self.times = np.linspace(0,1,T)
        self.rho = (1-self.times.reshape(spacetime_grid_shape))*mu + self.times.reshape(spacetime_grid_shape)*nu # space-time density
        self.m = np.zeros((self.d,) + spacetime_grid_shape) # space time vector field
        self.M = np.concatenate((self.rho[np.newaxis,...],self.m)) # the target unknown

        self.phi = np.zeros( mesh.shape + (T,)) # test function, wasserstein potential
        self.nabla_phi = np.zeros((self.d,1) + self.mesh.shape + (T,)) # space-time gradient of phi

        self.a = np.zeros(self.mesh.shape + (T,))
        self.b = np.zeros((self.d,)+ self.mesh.shape + (T,))
        self.c = np.concatenate((self.a[np.newaxis,...],self.b)) # parameters of the Benamou functional 

        self.tau=1

    def __str__(self):
        return  str(self.__class__) + '\n'+ '\n'.join(('{} = {}'.format(item, self.__dict__[item]) for item in self.__dict__))
    def poisson_step():
        # return the solution to tau*laplacian(phi) = gradient(tau*c-M)
        return
    def projection_step(self):
        alpha_beta = self.nabla_phi + self.M / self.tau
        alpha = alpha_beta[0]
        beta = np.linalg.norm(alpha_beta[1:], axis=0)
        f = lambda t: (alpha-0.5*t)(1+0.5*t)**2 + 0.5*v**2
        return

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

    def dual_step(self):
        self.M = self.M - self.tau*(self.c - self.nabla_phi)

        
