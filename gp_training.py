import numpy as np
import scipy as sp
import george
from george.kernels import ExpSquaredKernel
from george import kernels
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as op
#import emcee
from scipy.linalg import cholesky, cho_solve
from scipy.spatial import cKDTree as KDTree
from scipy import stats

class gp_tr(object):
    def __init__(self, x, y, yerr, gp, optimize=False, MCMC=False, save=False, savename='GP_mcmc'):
        self.x = x
        self.y = y
        self.yerr = yerr
        self.gp = gp
        self.p0 = gp.kernel.pars
        #self.gp.compute(self.x)
        
        if optimize == True and MCMC == True:
            print("Both optimization and MCMC are chosen, only optimization will run...")
            MCMC = False
        
        elif optimize == True and MCMC == False:
            self.bnd = [np.log((1e-6, 1e+6)) for i in range(len(self.p0))]
            self.results = op.minimize(self.nll, self.p0, jac=self.grad_nll, method='L-BFGS-B', bounds=self.bnd)
            self.gp.kernel[:] = self.results.x
            self.p_op = self.results.x
        
        elif optimize == False and MCMC == False:
            self.p_op = self.gp.kernel.vector
        
        elif optimize == False and MCMC == True:
            ndim = len(self.p0)
            nwalkers = 2*ndim
            Nstep = 5
            print(ndim, nwalkers)
            position = [self.p0+1e-3*np.random.randn(len(self.p0)) for jj in range(nwalkers)]
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob)
            sampler.run_mcmc(position, Nstep)
            samples  =sampler.chain[:,:,:].reshape((-1, ndim))
            chi2p = sampler.lnprobability[:,:].reshape(-1)
            
            mcmc_data = zip(-chi2p, samples)
            self.p_op1 = mcmc_data[np.where(mcmc_data[:,0]==min(mcmc_data[:,0]))]
            self.p_op = self.p_op[0]
            if save == True:
                np.savetxt(savename+"_mcmc.dat", mcmc_data, fmt = '%.6e')

    def nll(self, p):
        self.gp.kernel[:] = p
        ll = self.gp.lnlikelihood(self.y, quiet=True)
        return -ll if np.isfinite(ll) else 1e25
    
    def grad_nll(self, p):
        self.gp.kernel[:] = p
        return -self.gp.grad_lnlikelihood(self.y, quiet=True)

    def lnprob(self, p):
        if np.any((-5.0>p) + (p>5.0)):
            return -np.inf
        lnprior = 0.0
        #self.gp.compute(self.x)
        self.gp.kernel[:] = p
        return lnprior + self.gp.lnlikelihood(self.y, quiet=True)





