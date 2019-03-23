#   This code is used for prediction. The training process is performed in another code. So the output from the training process: the optimized hyper parameters of the Gaussian process, is now the input. Note the kernels in this prediction code and in the graining process should be exactly the same, as well as the input dataset.

import numpy as np
import scipy as sp
import george
from george.kernels import *
import sys
from gp_training import *
from time import time
import scipy.interpolate

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

pp = np.loadtxt("wp_covar_9bins_pp.dat")

HODLR = False

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

y2 = np.empty((len(rr)*9))
ss2 = 0
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
            y2[ss2] = y[ss]
            ss = ss+1
            ss2 = ss2+1

######

    p0 = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    
    k1 = ExpSquaredKernel(p0, ndim=len(p0))
    k2 = Matern32Kernel(p0, ndim=len(p0))
    k3 = ConstantKernel(0.1, ndim=len(p0))
    k4 = WhiteKernel(0.1, ndim=len(p0))
    k5 = ConstantKernel(0.1, ndim=len(p0))

    kernel = k1*k5+k2+k3+k4
    
    ppt = pp[j]

    if j == 0:
        if HODLR == True:
            gp0 = george.GP(kernel, mean=np.mean(y), solver=george.HODLRSolver)
        else:
            gp0 = george.GP(kernel, mean=np.mean(y), solver=george.BasicSolver)
        gp0.compute(rr, yerr)

        gp0.kernel.vector = ppt
        gp0.compute(rr, yerr)

    if j == 1:
        if HODLR == True:
            gp1 = george.GP(kernel, mean=np.mean(y), solver=george.HODLRSolver)
        else:
            gp1 = george.GP(kernel, mean=np.mean(y), solver=george.BasicSolver)
        gp1.compute(rr, yerr)
        
        gp1.kernel.vector = ppt
        gp1.compute(rr, yerr)

    if j == 2:
        if HODLR == True:
            gp2 = george.GP(kernel, mean=np.mean(y), solver=george.HODLRSolver)
        else:
            gp2 = george.GP(kernel, mean=np.mean(y), solver=george.BasicSolver)
        gp2.compute(rr, yerr)
        
        gp2.kernel.vector = ppt
        gp2.compute(rr, yerr)

    if j == 3:
        if HODLR == True:
            gp3 = george.GP(kernel, mean=np.mean(y), solver=george.HODLRSolver)
        else:
            gp3 = george.GP(kernel, mean=np.mean(y), solver=george.BasicSolver)
        gp3.compute(rr, yerr)

        gp3.kernel.vector = ppt
        gp3.compute(rr, yerr)

    if j == 4:
        if HODLR == True:
            gp4 = george.GP(kernel, mean=np.mean(y), solver=george.HODLRSolver)
        else:
            gp4 = george.GP(kernel, mean=np.mean(y), solver=george.BasicSolver)
        gp4.compute(rr, yerr)
        
        gp4.kernel.vector = ppt
        gp4.compute(rr, yerr)

    if j == 5:
        if HODLR == True:
            gp5 = george.GP(kernel, mean=np.mean(y), solver=george.HODLRSolver)
        else:
            gp5 = george.GP(kernel, mean=np.mean(y), solver=george.BasicSolver)
        gp5.compute(rr, yerr)
        
        gp5.kernel.vector = ppt
        gp5.compute(rr, yerr)

    if j == 6:
        if HODLR == True:
            gp6 = george.GP(kernel, mean=np.mean(y), solver=george.HODLRSolver)
        else:
            gp6 = george.GP(kernel, mean=np.mean(y), solver=george.BasicSolver)
        gp6.compute(rr, yerr)
        
        gp6.kernel.vector = ppt
        gp6.compute(rr, yerr)

    if j == 7:
        if HODLR == True:
            gp7 = george.GP(kernel, mean=np.mean(y), solver=george.HODLRSolver)
        else:
            gp7 = george.GP(kernel, mean=np.mean(y), solver=george.BasicSolver)
        gp7.compute(rr, yerr)
        
        gp7.kernel.vector = ppt
        gp7.compute(rr, yerr)

    if j == 8:
        if HODLR == True:
            gp8 = george.GP(kernel, mean=np.mean(y), solver=george.HODLRSolver)
        else:
            gp8 = george.GP(kernel, mean=np.mean(y), solver=george.BasicSolver)
        gp8.compute(rr, yerr)
        
        gp8.kernel.vector = ppt
        gp8.compute(rr, yerr)


