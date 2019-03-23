import numpy as np
import george
from george.kernels import *
import scipy.optimize as op
from scipy.linalg import cholesky, cho_solve
import sys
import gp_training
from gp_training import *

x = np.loadtxt("HOD_design_np8_n5000.dat")
x = x[0:2000]

x[:,0] = np.log10(x[:,0])
x[:,2] = np.log10(x[:,2])

xc = np.loadtxt("cosmology_camb_full.dat")

CC = range(0,40)

HH = np.array(range(0,2000))
HH  = HH.reshape(40, 50)
rr = np.empty((HH.shape[1]*len(CC), x.shape[1]+xc.shape[1]))
YY = np.empty((9, HH.shape[1]*len(CC)))


##################   find the mean of the data  #############


s2 = 0
for CID in CC:
    for HID in HH[CID]:
        HID = int(HID)

        d = np.loadtxt("training/wp_results/wp_cosmo_"+str(CID)+"_HOD_"+str(HID)+".dat")
        YY[:,s2] = d[:,1]
        s2 = s2+1

Ymean = np.mean(YY, axis=1)

##################  found the mean of the data ################

GP_err = np.loadtxt("training/wp_results/Cosmo_err.dat")

pp = []
for j in range(9):
    DC = j
    Ym = Ymean[DC]
    ss = 0
    yerr = np.zeros((len(rr)))
    y = np.empty((len(rr)))
    for CID in CC:
        for HID in HH[CID]:
            HID = int(HID)
            
            d = np.loadtxt("training/wp_results/wp_cosmo_"+str(CID)+"_HOD_"+str(HID)+".dat")
            
            rr[ss,0:7]=xc[CID, :]
            rr[ss,7:15]=x[HID, :]
                    
            d = d[:,1]
            d1 = d[DC]
            y[ss] = np.log10(d1/Ym)

            yerr[ss] = GP_err[j]/2.303
            ss = ss+1
#########

    p0 = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    k1 = ExpSquaredKernel(p0, ndim=len(p0))
    k2 = Matern32Kernel(p0, ndim=len(p0))
    k3 = ConstantKernel(0.1, ndim=len(p0))
    k4 = WhiteKernel(0.1, ndim=len(p0))
    k5 = ConstantKernel(0.1, ndim=len(p0))

    kernel = k1*k5+k2+k3+k4

    gp = george.GP(kernel, mean=np.mean(y), solver=george.BasicSolver)
    gp.compute(rr, yerr)

    ppt = gp_tr(rr, y, yerr, gp, optimize=True).p_op
    gp.kernel.vector = ppt
    gp.compute(rr, yerr)
    print(ppt)
    pp.append(ppt)


#np.savetxt(data+"_9bins_pp.dat", pp, fmt='%.7f')  # save the optimized hyperparameters

