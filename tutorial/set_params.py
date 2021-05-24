import sys
import os
import numpy as np
import pathlib


MPI_process = True
if MPI_process == False:
    print('you are not using MPI')

#___________________________________________________________________________________________________

#SIMS settings_________________________________________________________________________________
#___________________________________________________________________________________________________

n_params  = 2
theta_fiducial  = np.array([70,-0.5])

selection = True


n_obs=3


sorting = 'noisy redshift' # 'true redshift' or 'noisy redshift'

n_sources = 100

if selection:
    z_range = [0.0,0.13]
else:
    z_range = [0.0,0.05]
sigmaz = 0.001
interpolation_bins = 100              #for redshift CDF computation

meanv = 0
sigmavlos = 500
sigmav = 200 

#constants
c = 299792
G = 4.3*10**(-9)

####################################################################################################

ref_filename = '_3D'
    
if selection == True:
    ref_filename += '_selection'
    
if sorting == 'true redshift':
    ref_filename += '_truesort'
else:
    ref_filename += '_noisysort'
####################################################################################################

#STAN settings______________________________________________________________________________________
#___________________________________________________________________________________________________

n_coeff_fit = 15
grid_bins = 10


#___________________________________________________________________________________________________ 
   
#NN settings______________________________________________________________________________________
#___________________________________________________________________________________________________

draws_method = 1                    # 1 = latin hypercube, 2 = uniform distribution

N_draws = 1000

n_s  = 5                          # sets of sims for training
n_s_val = 2                         # sets of sims for validation 

n_members = 5


#architecture
loss_function = np.array(['MSE'])#,'MAE'])
bs = np.array([100,500])
lr = np.array([0.0001,0.0005,0.001])
l2 = np.array([0,0.0001,0.0002])
l1 = np.array([0.0001,0.0002])

#___________________________________________________________________________________________________ 

#PYDELFI settings___________________________________________________________________________________
#___________________________________________________________________________________________________

q0_prior=True
using_prerun_sims = True   #for the inference
NDE_training_epochs = 500

lower = np.array([60,-2])
upper = np.array([80,1])

n_mds = 5

#plotting_single_NDES = True


#___________________________________________________________________________________________________

#FILENAMES FOR DATA STORAGE_________________________________________________________________________
#___________________________________________________________________________________________________


main_data_folder = './datasets'

pathtoNNsims = '{}/NN/'.format(main_data_folder)
pathtoNNmodels = '../NN_results/'
pathtoLFIresults = '../LFI_results/'

pathtorealsamples = '../{}/real_samples'.format(main_data_folder) 
pathtoSTANresults = 'STAN_results/multi_rnd_%s'%n_obs+'D'
if selection == True:
    pathtoSTANresults += '_selection'

pathtopydelfiprerunsims = '{}/pydelfi_prerun_sims'.format(main_data_folder)

