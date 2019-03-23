import numpy as np
import scipy as sp
import george
from george.kernels import *
import sys
from gp_training import *
from time import time
import scipy.interpolate
import gp_wp_cos_hod_9bins_init2 as gw
import gp_mono_cos_hod_9bins_init2 as gm
import gp_quad_cos_hod_9bins_init2 as gq
import scipy.linalg as la


def wp_prediction(param):
    # the prediction of wp is wp(rp)

    tc = np.atleast_2d(param)
    
    wp_mu0, wp_cov0 = gw.gp0.predict(gw.y2[len(gw.CC)*gw.HH.shape[1]*0:len(gw.CC)*gw.HH.shape[1]*1], tc)
    wp_mu1, wp_cov1 = gw.gp1.predict(gw.y2[len(gw.CC)*gw.HH.shape[1]*1:len(gw.CC)*gw.HH.shape[1]*2], tc)
    wp_mu2, wp_cov2 = gw.gp2.predict(gw.y2[len(gw.CC)*gw.HH.shape[1]*2:len(gw.CC)*gw.HH.shape[1]*3], tc)
    wp_mu3, wp_cov3 = gw.gp3.predict(gw.y2[len(gw.CC)*gw.HH.shape[1]*3:len(gw.CC)*gw.HH.shape[1]*4], tc)
    wp_mu4, wp_cov4 = gw.gp4.predict(gw.y2[len(gw.CC)*gw.HH.shape[1]*4:len(gw.CC)*gw.HH.shape[1]*5], tc)
    wp_mu5, wp_cov5 = gw.gp5.predict(gw.y2[len(gw.CC)*gw.HH.shape[1]*5:len(gw.CC)*gw.HH.shape[1]*6], tc)
    wp_mu6, wp_cov6 = gw.gp6.predict(gw.y2[len(gw.CC)*gw.HH.shape[1]*6:len(gw.CC)*gw.HH.shape[1]*7], tc)
    wp_mu7, wp_cov7 = gw.gp7.predict(gw.y2[len(gw.CC)*gw.HH.shape[1]*7:len(gw.CC)*gw.HH.shape[1]*8], tc)
    wp_mu8, wp_cov8 = gw.gp8.predict(gw.y2[len(gw.CC)*gw.HH.shape[1]*8:len(gw.CC)*gw.HH.shape[1]*9], tc)
    
    wp_mu = list(wp_mu0)+list(wp_mu1)+list(wp_mu2)+list(wp_mu3)+list(wp_mu4)+list(wp_mu5)+list(wp_mu6)+list(wp_mu7)+list(wp_mu8)
    wp_mu = np.array(wp_mu)
    wp_pre=10**wp_mu*gw.Ymean

    return wp_pre

def mono_prediction(param):
    # the prediction of monopole is \xi_{0}(s)

    tc = np.atleast_2d(param)

    mo_mu0, mo_cov0 = gm.gp0.predict(gm.y2[len(gm.CC)*gm.HH.shape[1]*0:len(gm.CC)*gm.HH.shape[1]*1], tc)
    mo_mu1, mo_cov1 = gm.gp1.predict(gm.y2[len(gm.CC)*gm.HH.shape[1]*1:len(gm.CC)*gm.HH.shape[1]*2], tc)
    mo_mu2, mo_cov2 = gm.gp2.predict(gm.y2[len(gm.CC)*gm.HH.shape[1]*2:len(gm.CC)*gm.HH.shape[1]*3], tc)
    mo_mu3, mo_cov3 = gm.gp3.predict(gm.y2[len(gm.CC)*gm.HH.shape[1]*3:len(gm.CC)*gm.HH.shape[1]*4], tc)
    mo_mu4, mo_cov4 = gm.gp4.predict(gm.y2[len(gm.CC)*gm.HH.shape[1]*4:len(gm.CC)*gm.HH.shape[1]*5], tc)
    mo_mu5, mo_cov5 = gm.gp5.predict(gm.y2[len(gm.CC)*gm.HH.shape[1]*5:len(gm.CC)*gm.HH.shape[1]*6], tc)
    mo_mu6, mo_cov6 = gm.gp6.predict(gm.y2[len(gm.CC)*gm.HH.shape[1]*6:len(gm.CC)*gm.HH.shape[1]*7], tc)
    mo_mu7, mo_cov7 = gm.gp7.predict(gm.y2[len(gm.CC)*gm.HH.shape[1]*7:len(gm.CC)*gm.HH.shape[1]*8], tc)
    mo_mu8, mo_cov8 = gm.gp8.predict(gm.y2[len(gm.CC)*gm.HH.shape[1]*8:len(gm.CC)*gm.HH.shape[1]*9], tc)
        
    mo_mu = list(mo_mu0)+list(mo_mu1)+list(mo_mu2)+list(mo_mu3)+list(mo_mu4)+list(mo_mu5)+list(mo_mu6)+list(mo_mu7)+list(mo_mu8)
    mo_mu = np.array(mo_mu)
    mo_pre=10**mo_mu*gm.Ymean

    return mono_pre

def quad_prediction(param):
    # the prediction of quadrupole is \xi_{2}(s)*s^2, it's different than monopole. devide by s^2 will give \xi_{2}, the value of s can be found from any training or test sample.

    tc = np.atleast_2d(param)

    qu_mu0, qu_cov0 = gq.gp0.predict(gq.y2[len(gq.CC)*gq.HH.shape[1]*0:len(gq.CC)*gq.HH.shape[1]*1], tc)
    qu_mu1, qu_cov1 = gq.gp1.predict(gq.y2[len(gq.CC)*gq.HH.shape[1]*1:len(gq.CC)*gq.HH.shape[1]*2], tc)
    qu_mu2, qu_cov2 = gq.gp2.predict(gq.y2[len(gq.CC)*gq.HH.shape[1]*2:len(gq.CC)*gq.HH.shape[1]*3], tc)
    qu_mu3, qu_cov3 = gq.gp3.predict(gq.y2[len(gq.CC)*gq.HH.shape[1]*3:len(gq.CC)*gq.HH.shape[1]*4], tc)
    qu_mu4, qu_cov4 = gq.gp4.predict(gq.y2[len(gq.CC)*gq.HH.shape[1]*4:len(gq.CC)*gq.HH.shape[1]*5], tc)
    qu_mu5, qu_cov5 = gq.gp5.predict(gq.y2[len(gq.CC)*gq.HH.shape[1]*5:len(gq.CC)*gq.HH.shape[1]*6], tc)
    qu_mu6, qu_cov6 = gq.gp6.predict(gq.y2[len(gq.CC)*gq.HH.shape[1]*6:len(gq.CC)*gq.HH.shape[1]*7], tc)
    qu_mu7, qu_cov7 = gq.gp7.predict(gq.y2[len(gq.CC)*gq.HH.shape[1]*7:len(gq.CC)*gq.HH.shape[1]*8], tc)
    qu_mu8, qu_cov8 = gq.gp8.predict(gq.y2[len(gq.CC)*gq.HH.shape[1]*8:len(gq.CC)*gq.HH.shape[1]*9], tc)
    qu_mu = list(qu_mu0)+list(qu_mu1)+list(qu_mu2)+list(qu_mu3)+list(qu_mu4)+list(qu_mu5)+list(qu_mu6)+list(qu_mu7)+list(qu_mu8)
    qu_mu = np.array(qu_mu)
    qu_pre=qu_mu*gq.Ymean

    return quad_prediction



