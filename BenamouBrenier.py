''' Implementation of the Benamou_Brenier method based on fluid dynamics.
'''
import numpy as np
import warnings
from warnings import warn

class TransportProblem:
    def __init__(self,mesh,mu,nu,T,tau=1):
        """
        With N = N_1*...*N_d discretization
        Inputs:
                mesh: array of shape            (d,N_1,...,N_d)
                mu,nu: arrays of shape            (N_1,...,N_d)
                T: integer
        Initialized:
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
        (d,*space_grid_shape) = mesh.shape
        space_grid_shape = tuple(space_grid_shape)
        if space_grid_shape != mu.shape or space_grid_shape != nu.shape:
            Exception("The space grid dimensions of the mesh doesn't match the mesures")
        spacetime_grid_shape = (T,) + space_grid_shape # (T, N_1,...,N_d)

        self.spacetime_grid_shape = spacetime_grid_shape
        self.d = d
        self.mesh = mesh # mesh for the measure discretization
        self.mu = mu # source measure
        self.nu = nu # target measure

        self.T = T
        self.times = np.linspace(0,1,T)
        self.rho = (1-self.times.reshape((T,)+d*(1,)))*mu + self.times.reshape((T,)+d*(1,))*nu # space-time density
        self.m = np.zeros((self.d,) + spacetime_grid_shape) # space time vector field
        self.M = np.concatenate((self.rho[np.newaxis,...], self.m)) # the target unknown

        self.phi = np.zeros(spacetime_grid_shape) # test function, wasserstein potential
        self.nabla_phi = np.zeros((d+1,) + spacetime_grid_shape) # space-time gradient of phi

        self.a = np.zeros(spacetime_grid_shape)
        self.b = np.zeros((d,)+ spacetime_grid_shape)
        self.c = np.concatenate((self.a[np.newaxis,...], self.b)) # NEED UPDATE # parameters of the Benamou functional 

        self.tau=tau

    def __str__(self):
        "print class properties"
        return  str(self.__class__) + '\n'+ '\n'.join(('{} = {}'.format(item, self.__dict__[item]) for item in self.__dict__))
    
    def update_c(self):
         self.c = np.concatenate((self.a[np.newaxis,...], self.b))
    def update_rho_m(self):
        self.rho = self.M[0]
        self.m = self.M[1:]

    def poisson_step():
        # return the solution to tau*laplacian(phi) = div(tau*c-M)
        return
    
    def projection_step(self):
        alpha_beta = self.nabla_phi + self.M / self.tau
        tol = 1e-5
        if np.max(alpha_beta) <= 0+tol: # already in the set
            return
        
        maxiter = 1000 # TODO: to be tuned
        nbr_maxiter_reached = 0
        # iter on the grid:
        for index in np.ndindex(self.spacetime_grid_shape): # grid-wise
            alpha = alpha_beta[0][index]
            beta = np.linalg.norm(alpha_beta[1:][(...,*index)], axis=0) # make it a 2D problem
            f = lambda t: (alpha-0.5*t)*(1+0.5*t)**2 + 0.5*beta**2
            df = lambda t: (0.5*t+1)*(-0.75*t+alpha-0.5) # derivative

            t = (alpha-0.5)*4/3 + 100 # initialize t so that it is far from the local max
            i=0
            # Newton method:
            while np.abs(f(t)) > tol and i < maxiter:
                t = t - f(t)/df(t) # newton step
                i=i+1
            if i==maxiter:
                nbr_maxiter_reached = nbr_maxiter_reached + 1
                warn("Max number of iterations reached in Newton's method.")
            # Update a and b
            self.a[index] = alpha - 1/2*t # update a
            self.b[(...,*index)] = alpha_beta[1:][(...,*index)]/(1/2*t+1) # update b
            self.update_c()
            assert self.a[index] + 1/2*np.sum(alpha_beta[1:][(...,*index)]**2, axis=0) <= 0 + tol
            self.a[index] = - 1/2*np.sum(alpha_beta[1:][(...,*index)]**2, axis=0)

        if nbr_maxiter_reached > 0:
            warn("Max number of iterations reached in Newton's method ("+str(nbr_maxiter_reached)+" times).")
        else:
            print("Projection step converged to tolerance.")
        return
    
    def dual_step(self):
        self.M = self.M - self.tau*(self.c - self.nabla_phi)
        self.update_rho_m()
        print("Dual step done.")

    def k(rho,m): # uselfull?
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



        
