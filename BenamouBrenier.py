''' Implementation of the Benamou_Brenier method based on fluid dynamics.
'''
import numpy as np
import matplotlib.pyplot as plt
import warnings
from time import time
from warnings import warn
from tqdm.notebook import tqdm,trange # to display loading bars
from matplotlib.widgets import Slider
from poisson import laplacian_matrix,poisson,derivative_matrices,divergence,gradient

class TransportProblem:
    def __init__(self,mesh,mu,nu,T,tau=1,display=True):
        """
        With N = N_1*...*N_d discretization
        Inputs:
                mesh: array of shape            (d,N_1,...,N_d)     mesh grid that discretize space.
                mu,nu: arrays of shape            (N_1,...,N_d)     source and target measure of transportation .
                T: integer                                          number of discretized times.
        Initialized:
                times: array of shape          (T,)                 array of the discretized times
                rho: array of shape            (T, N_1,...,N_d)     pressure.
                m: array of shape           (d, T, N_1,...,N_d)     pressure*velocity field
                M: array of shape         (1+d, T, N_1,...,N_d)     =(rho,m)
                phi: array of shape            (T, N_1,...,N_d)     Lagrange multiplier
                nabla_phi: array of shape (1+d, T, N_1,...,N_d)     gradient of phi
                a: array of shape              (T, N_1,...,N_d)     parameter of the Benamou functional
                b: array of shape           (d, T, N_1,...,N_d)     parameter of the Benamou functional
                c: array of shape         (1+d, T, N_1,...,N_d)     =(a,b)

        """
        # creation of shapes tuples, last dimensions always contain space and time:
        (d,*space_grid_shape) = mesh.shape
        if d!=2:
            warn(str(d)+"D space, poisson step not implemented.")
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
        # rho inizialized at L2 interpolation (pointwise):
        self.rho = (1-self.times.reshape((T,)+d*(1,)))*mu + self.times.reshape((T,)+d*(1,))*nu # space-time density
        eps = 0.1
        self.rho = (1-eps)*self.rho + eps/np.prod(spacetime_grid_shape)
        self.m = np.zeros((self.d,) + spacetime_grid_shape) # space time vector field
        self.M = np.concatenate((self.rho[np.newaxis,...], self.m)) # the target unknown

        self.phi = np.zeros(spacetime_grid_shape) # test function, wasserstein potential
        self.nabla_phi = np.zeros((d+1,) + spacetime_grid_shape) # space-time gradient of phi
        self.laplacian_matrix = laplacian_matrix(space_grid_shape[0])
        self.An, self.Ap = derivative_matrices(space_grid_shape[0])

        self.a = np.zeros(spacetime_grid_shape)
        self.b = np.zeros((d,)+ spacetime_grid_shape)
        self.c = np.concatenate((self.a[np.newaxis,...], self.b)) # NEED UPDATE # parameters of the Benamou functional 

        self.tau=tau
        if display:
            print("TransportProblem object initialized.")

    def __str__(self):
        " Print all class properties. "
        return  str(self.__class__) + '\n'+ '\n'.join(('{} = {}'.format(item, self.__dict__[item]) for item in self.__dict__))
    
    def update_c(self):
         " Update c according to a and b. "
         self.c = np.concatenate((self.a[np.newaxis,...], self.b))

    def update_rho_m(self):
        " Update rho and m according to M. "
        self.rho = self.M[0]
        self.m = self.M[1:]

    def poisson_step(self,display=False):
        """  Compute the solution to tau*laplacian(phi) = div(tau*c-M) ,
        with time-Neumann conditions    tau * d/dt phi(0,.) = mu - rho(0,.) + tau * a (0,.)
                                        tau * d/dt phi(1,.) = nu - rho(1,.) + tau * a (1,.) """
        if self.d!=3:
            Exception("Poisson step is implemented only for a 2D space, not for a "
                      + str(self.d) + "D space.")
        g = np.stack((self.mu, self.nu))/self.tau - self.rho[[0,-1]]/self.tau + self.a[[0,-1]]
        f = divergence(self.c-self.M/self.tau, self.An, self.Ap)
        self.phi,_,_,_ = poisson(f,g,self.laplacian_matrix)
        self.nabla_phi = gradient(self.phi,self.An,self.Ap)
        if display:
            print("Poisson step done.")
    
    def projection_step(self,display=False):
        " Minimize c, computed as the orthogonal projection of nabla_phi+M/tau-c to the parabola K = {(a,b) st. a+|b|^2 < 0}. "
        a=self.a
        b=self.b
        alpha_beta = self.nabla_phi + self.M / self.tau
        tol = 1e-8
        if np.max(alpha_beta) <= 0+tol: # already in the set
            return
        f = lambda t,alpha,beta: (alpha-0.5*t)*(1+0.5*t)**2 + 0.5*beta**2 # function to find zero
        df = lambda t,alpha: (0.5*t+1)*(-0.75*t+alpha-0.5) # derivative of f
        maxiter = 50 # TODO: to be tuned
        nbr_maxiter_reached = 0
        # iter on the grid:
        iterator = np.ndindex(self.spacetime_grid_shape)
        if display:
            iterator = tqdm(iterator,total=np.prod(self.spacetime_grid_shape))
        for index in iterator: # grid-wise
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
        elif display:
            print("Projection step converged to tolerance.")
        return
    
    def projection_step_bis(self, display=False):
        alpha_beta = self.nabla_phi + self.M / self.tau
        alpha,beta = alpha_beta[0], np.sqrt(np.sum(alpha_beta[1:]**2,axis=0))
        iterator = np.ndindex(self.spacetime_grid_shape)
        if display:
            iterator = tqdm(iterator,total=np.prod(self.spacetime_grid_shape))
        for index in iterator: # grid-wise
            if np.max(alpha[index] + beta[index]**2/2,) > 0: # already in the set
                a,b,c = 4-2*alpha[index], 4-8*alpha[index], 4*beta[index]**2-8*alpha[index]
                t = last_root(a,b,c)
                self.a[index] = alpha[index] - 1/2*t
                self.b[(...,*index)] = alpha_beta[1:][(...,*index)]/(1/2*t+1)
        self.update_c()
        if display:
            print("Projection step done.")

    def dual_step(self,display=False):
        " Compute the dual step, a gradient step of the dual variable M."
        self.M = self.M - self.tau*(self.c - self.nabla_phi)
        self.update_rho_m()
        if display:
            print("Dual step done.")

    def residual(self):
        " Compute the residual of the Hamilton-Jacobi equation "
        return self.nabla_phi[0] + 0.5 * np.sum(self.nabla_phi[1:]**2,axis=0)
    
    def criterium(self):
        " Normalized residual "
        try:
            return np.sum(self.rho * np.abs(self.residual())) / np.sum(self.rho * np.sum(self.nabla_phi[1:]**2,axis=0))
        except ZeroDivisionError:
            warn("Division by zero in criterium (rho * nabla_phi = 0), infinity returned.")
            return np.inf
        
    def solve(self,tol=1e-7,maxiter=100,display=True):
        """ Proceed augmented Lagrandgian method by doing iteratively the three steps poisson-projection-dual,
        until convergence is detected by residual or maxiter. """
        criteria=[]
        iterator = range(maxiter)
        if display:
            iterator = tqdm(iterator)
        for i in iterator:
            self.poisson_step()
            self.projection_step_bis()
            self.dual_step()
            crit = self.criterium()
            res = np.max(np.abs(self.residual()))
            criteria.append(crit)
            if res < tol:
                break
        if res < tol and display:
            print("Benamou-Brenier method converged to tolerance with criterium = "+str(crit))
        elif display:
           print("Benamou-Brenier method stopped at the maximum number of iterations, with criterium = "+str(crit)+">"+str(tol))
        return criteria
    
    def plot(self,t=None):
        " plot rho[i] with a slider for i. Use <%matplotlib>, and turn back to <%matplotlib inline> after. "
        if t is None:
            fig, (ax1,ax2) = plt.subplots(2)
            self.s = Slider(ax = ax2, label = 'value', valmin = 0, valmax = self.T-1, valinit = 1)
            def update(val):
                value=int(self.s.val)
                ax1.cla()
                ax1.contour(self.rho[value])
            self.s.on_changed(update)
            update(0)
            plt.show()
            return fig
        else:
            tt = np.atleast_1d(t)
            for i,t in enumerate(tt):
                plt.figure()
                plt.contour(np.arange(self.T)/(self.T-1),np.arange(self.T)/(self.T-1),self.rho[int(t*(self.T-1))])
                plt.title("t="+str(t))
                plt.colorbar()
                plt.savefig("graphics/rho("+str(i)+").pdf")

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
        
def last_root(a,b,c):
    p = b - a**2/3
    q = a / 27 * (2*a**2 - 9*b) + c
    delta = (p/3)**3 + (q/2)**2
    if delta>0:
        u = np.cbrt(-q/2 + np.sqrt(delta))
        v = np.cbrt(-q/2 - np.sqrt(delta))
        x = u + v - a/3
    elif delta == 0:
        u = np.cbrt(-q/2)
        x = 2*np.abs(u) - a/3
    else:
        u = (-q/2 + 1j*np.sqrt(-delta))**(1/3)
        x = 2*np.real(u) - a/3
    return x
