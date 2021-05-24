from __future__ import absolute_import, division, print_function, unicode_literals

import os.path

import sys

import numpy as np
import pandas as pd

#useful for single NDEs plots__________________________
from matplotlib import cm
import matplotlib.cm as mpcm
from getdist import plots, MCSamples
#______________________________________________________
import pickle as pkl

import matplotlib.pyplot as plt

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import tensorflow as tf
from tensorflow.keras.models import load_model

print("tensorflow version", tf.__version__)

#importing simulation modules and parameters file______
sys.path.insert(1, '../')
from set_params import *
import simulator as sims
data_generation=sims.simulations()
#______________________________________________________


z_min = z_range[0]

if selection == False:
    sm_BNS = pkl.load(open('STAN_scripts/3D_model.pkl', 'rb'))
else:
    sm_BNS = pkl.load(open('STAN_scripts/3Dsel_model.pkl', 'rb'))
    numerical_fit = np.loadtxt('STAN_scripts/N_bar_numerical_coefficients_3Dsel_zmin{}.txt'.format(z_min))
    print('STAN_scripts/N_bar_numerical_coefficients_3Dsel_zmin{}.txt'.format(z_min))

if  selection==True:
    N_coeff = np.zeros((n_coeff_fit))
    for i in range (N_coeff.shape[0]):
        N_coeff[i] = numerical_fit[i]
    

if os.path.isfile('real_data_element.npy'):
    index = int(np.load('real_data_element.npy'))
else:
    index = 0
    
real_data = np.loadtxt('../{}'.format(pathtorealsamples)+'/multi_randomdata{}.txt'.format(ref_filename))

redshifts = real_data[:100]
distances = real_data[100:200]
v = real_data[200:300]
        

if selection == False:
    BNS_data = {'J': n_sources,
            'z_obs': redshifts,
            'D_obs': distances,
            'frac_err_D': 0.1,
            'v_obs':v,
            'mean_v':meanv,
            'sigma_vlos':sigmavlos,
            'sigma_z': sigmaz,
            'sigma_v': sigmav,       
            'zmax': z_range[1],
            'zmin': z_range[0]}

else:
    BNS_data = {'J': n_sources,
            'z_obs': redshifts,
            'D_obs': distances,
            'frac_err_D': 0.1,
            'v_obs':v,
            'mean_v':meanv,
            'sigma_vlos':sigmavlos,
            'sigma_z': sigmaz,
            'sigma_v': sigmav,       
            'zmax': z_range[1],
            'zmin': z_range[0],
            'N': n_coeff_fit,      
            'numerical_coefficients':N_coeff}  
        

        
n_iter=10000

fit_BNS = sm_BNS.sampling(data=BNS_data, iter=n_iter, chains=4)

H = fit_BNS.extract()['H']
q = fit_BNS.extract()['q']

STAN_samples = np.column_stack((H,q))


if not os.path.exists('{}'.format(pathtoSTANresults)):
    os.makedirs('{}'.format(pathtoSTANresults))  
    os.makedirs('{}'.format(pathtoSTANresults)+'/posteriors') 
    
file_stan = '{}/'.format(pathtoSTANresults)+'multirnd_STAN_samples.npy'

if os.path.isfile(file_stan):
    old = np.load(file_stan)
    if (index==1) and (STAN_samples.shape[0]==old.shape[0]):
        new = np.stack((old,STAN_samples))
    elif (index>1) and (STAN_samples.shape[0]==old.shape[1]):
        new = np.concatenate((old,STAN_samples.reshape(1,STAN_samples.shape[0],STAN_samples.shape[1])))
    else:
        diff_dim = int(old.shape[1]-STAN_samples.shape[0])
        nan_array = np.empty((diff_dim,2))
        nan_array[:] = np.nan
        STAN_samples = np.row_stack((STAN_samples,nan_array))
        new = np.concatenate((old,STAN_samples.reshape(1,STAN_samples.shape[0],STAN_samples.shape[1])))
    print(index,new.shape)
    np.save(file_stan,new)
else:
    print(index,STAN_samples.shape)
    np.save(file_stan,STAN_samples)


names = ["H_0","q_0"]
labels = ["H_0","q_0"]

STAN_samples = MCSamples(samples=[STAN_samples], names = names, labels = labels, \
                    label='STAN' +  ' [$H_0$,$q_0$] = [%s ' %round(np.mean(H),2) + 
                       '$\pm$%s' %round(np.std(H),2) + ',%s'
                       %round(np.mean(q),2)
                       + '$\pm$%s' %round(np.std(q),2) +']')

samples_for_plot = [STAN_samples]

g = plots.get_subplot_plotter(width_inch = 12)
g.settings.alpha_filled_add = 0.6
g.triangle_plot(samples_for_plot, filled=True, normalized=True)
plt.legend()
plt.savefig('{}'.format(pathtoSTANresults)+'/posteriors/posterior_{}'.format(index) + '{}.png'.format(ref_filename))
