''' Implementation of the Benamou_Brenier method based on fluid dynamics.
'''
import numpy as np
import matplotlib.pyplot as plt
import warnings
from time import time
from warnings import warn
from tqdm.notebook import tqdm # to display loading bars

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
        # rho inizialized at L2 interpolation
        self.m = np.zeros((self.d,) + spacetime_grid_shape) # space time vector field
        self.M = np.concatenate((self.rho[np.newaxis,...], self.m)) # the target unknown

        self.phi = np.zeros(spacetime_grid_shape) # test function, wasserstein potential
        self.nabla_phi = np.zeros((d+1,) + spacetime_grid_shape) # space-time gradient of phi

        self.a = np.zeros(spacetime_grid_shape)
        self.b = np.zeros((d,)+ spacetime_grid_shape)
        self.c = np.concatenate((self.a[np.newaxis,...], self.b)) # NEED UPDATE # parameters of the Benamou functional 

        self.tau=tau
        print("TransportProblem object initialized.")

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
        a=self.a
        b=self.b
        alpha_beta = self.nabla_phi + self.M / self.tau
        tol = 1e-8
        if np.max(alpha_beta) <= 0+tol: # already in the set
            return
        f = lambda t,alpha,beta: (alpha-0.5*t)*(1+0.5*t)**2 + 0.5*beta**2
        df = lambda t,alpha: (0.5*t+1)*(-0.75*t+alpha-0.5) # derivative
        maxiter = 50 # TODO: to be tuned
        nbr_maxiter_reached = 0
        # iter on the grid:
        for index in tqdm(np.ndindex(self.spacetime_grid_shape),total=np.prod(self.spacetime_grid_shape)): # grid-wise
            temps = time()
            alpha = alpha_beta[0][index]
            beta = np.linalg.norm(alpha_beta[1:][(...,*index)], axis=0) # make it a 2D problem

            t = (alpha-0.5)*4/3 + 100 # initialize t so that it is far from the local max
            i=0
            im_f = f(t,alpha,beta)
            # Newton method:
            #print(time()-temps,end=" ")
            #temps = time()
            while np.abs(im_f) > tol and i < maxiter:
                t = t - im_f/df(t,alpha) # newton step
                im_f = f(t,alpha,beta)
                i=i+1
            #print(time()-temps,end=" ")
            #temps = time()
            if i==maxiter:
                nbr_maxiter_reached = nbr_maxiter_reached + 1
                warn("Max number of iterations reached in Newton's method.")
            # Update a and b
            a[index] = alpha - 1/2*t # update a
            b[(...,*index)] = alpha_beta[1:][(...,*index)]/(1/2*t+1) # update b
            # assert self.a[index] + 1/2*np.sum(alpha_beta[1:][(...,*index)]**2, axis=0) <= 0 + tol
            a[index] = - 1/2*np.sum(alpha_beta[1:][(...,*index)]**2, axis=0)
           # print(time()-temps)
        self.a=a
        self.b=b
        self.update_c()

        if nbr_maxiter_reached > 0:
            warn("Max number of iterations reached in Newton's method ("+str(nbr_maxiter_reached)+" times).")
        else:
            print("Projection step converged to tolerance.")
        return
    
    def dual_step(self):
        self.M = self.M - self.tau*(self.c - self.nabla_phi)
        self.update_rho_m()
        print("Dual step done.")

    def residual(self):
        return self.nabla_phi[0] + 0.5 * np.sum(self.nabla_phi[1:]**2,axis=0)
    def criterium(self):
        try:
            return np.sum(self.rho * self.residual()) / np.sum(self.rho * np.sum(self.nabla_phi[1:]**2,axis=0))
        except ZeroDivisionError:
            warn("Division by zero in criterium (rho * nabla_phi = 0), infinity returned.")
            return np.inf
    def solve(self,tol=1e-5,maxiter=100):
        i = 0
        condition = i < maxiter
        while condition:
            self.poisson_step()
            self.projection_step()
            self.dual_step()
            i = i + 1
            condition = self.criterium < tol and iter < maxiter
        crit = self.criterium()
        if crit < tol:
            print("Benamou-Brenier method converged to tolerance.")
        else:
           print("Benamou-Brenier method stopped at the maximum number of iterations, with criterium =", crit, ">",tol ,".")
        return
    
    def plot(self,i_time):
        fig = plt.contour(self.rho[i_time])
        plt.show()
        return fig


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



        
