# emulator
This repo has all the code and data for the emulator of galaxy correlation function, a part of the Aemulus project. The details of the project can be found from this website:

https://aemulusproject.github.io/index.html

This repo is for 

The Aemulus Project III: Emulation of the Galaxy Correlation Function

The details and usage of the code and data are as follows:

cosmology_camb_full.dat: the cosmologies of the training set

cosmology_camb_test_box_full.dat: the cosmologies of the test set

The columns are 

Omega_m, Omega_b, sigma_8, h, n_s, N_eff, w

The cosmology is based on wCDM model, and details of the simulations can be found from the above link and paper: Aemulus I

HOD_design_np8_n5000.dat: the HOD designs of the training set

HOD_test_np8_n1000.dat: the HOD designs of the test set

The training set has 5000 HOD designs, but only 2000 are used to build the emulator, since this number is able to saturate the parameter space. The test has 1000 HOD designs, but only the first 100 are used to evalulate the emulator performance. These two data file have the same format:

M_sat, alpha, M_cut, sigma_logM, v_bc, v_bs, c_vir, f

The definition and ranges of these parameters can be found from the paper: Aemulus III

gp_wp_cos_hod_9bins.py

gp_mono_cos_hod_9bins.py

gp_quad_cos_hod_9bins.py

These three scripts optimize the emulator for wp, RSD monopole and quadrupole respectively. You can build your own Gaussian process with different kernel functions, and optimize it.

I also provide the optimized hyper parameters for them:

wp_covar_9bins_pp.dat

RSD_multiple_mono_9bins_pp.dat

RSD_multiple_quad_9bins_pp.dat

You can directly use these optimized hyperparameter for the emulator and reproduce the results on our paper. The usage of these optimized emulator has examples as 

gp_wp_cos_hod_9bins_init2.py

gp_mono_cos_hod_9bins_init2.py

gp_quad_cos_hod_9bins_init2.py

You can import these three scripts to make predictions of galaxy correlation function for an arbitrary cosmology and HOD parameter set. The example is here:

gp_cos_hod_9bins_prediction.py

The initialization of this file may take a few seconds, depending on machine.

All the data for training and testing are saved in the file:

Archive.zip


--------------

This emulator is built on George, version 0.2.1, 

https://github.com/dfm/george

If you use the different version, the format of the kernel can be different and cause error.

---------------

If you have any questions or suggestions, let me know
