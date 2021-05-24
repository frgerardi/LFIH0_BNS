#!usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import glob

import numpy as np
import itertools
import pickle
from termcolor import colored, cprint

from decimal import *

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import signal  
import pathlib

#modules for NN_________________________________

from sklearn.metrics import mean_absolute_error 
from matplotlib import pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np
from math import*
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten

sys.path.insert(1, '../')
from set_params import *
import simulator as sims 
data_generation=sims.simulations()

if MPI_process == True:
    from mpi4py import MPI
else:
    print("you are not using MPI")

getcontext().prec = 3


# AUXILIARY FUNCTIONS________________________________________________________________________________

def normalization_function(data, include_theta = True,verbose=0):

    output = np.zeros((data.shape))

    norm_file = Path("../{}".format(main_data_folder)+"/norm_qnts{}.txt".format(ref_filename))
    if norm_file.is_file():
        if verbose == 1:
            print('WARNING: using previously computed normalization')
        norm_qnts = np.loadtxt(norm_file)
        if verbose == 1:
            print(norm_qnts.shape)
    else:
        if verbose == 1:
            print('WARNING: YOU HAVE NOT YET COMPUTED NORMALIZATION, YOU MIGHT NEED TO RERUN NN')
        
    for i in range (n_obs*n_sources):
        output[i] = (data[i]-norm_qnts[i,0])/norm_qnts[i,1]   
        
    
    if include_theta == True :
        for m in range (2):
            output[n_obs*n_sources+m] = (data[n_obs*n_sources+m]-lower[m])/(upper[m]-lower[m])
        
    return output


# MAIN BODY___________________________________________________________________________________________


if MPI_process == True:
    
    sendbuf=[]
    comm=MPI.COMM_WORLD
    rank=comm.rank
    size=comm.size

    m1,m2,m3 = np.meshgrid(l1,bs,lr)

    temp = itertools.count(0)
    rank_grid=np.array([[[next(temp) for i in range (lr.shape[0])] for j in range (l1.shape[0])] for k in range (bs.shape[0])])

    i = np.where(rank_grid==rank)[0]
    j = np.where(rank_grid==rank)[1]
    k = np.where(rank_grid==rank)[2]

    data=[m1[i,j,k],m2[i,j,k],m3[i,j,k]]

    l1 = data[0][0]
    bs = data[1][0]
    lr = data[2][0]
    
else:
    
    l1 = l1[0]
    bs = bs[0]
    lr = lr[0]
    
    data = [bs,lr,l1]


normalization_method = 2
NN_settings = '_2LR0.1_128_128'
training_data = N_draws*n_s

loss_funct = str(loss_function[0])
#print(loss_funct)

NN_filename = loss_funct +'_ns%s'%training_data + NN_settings + '_bs%s'%bs + '_lr%s'%lr + '_l1%s'%l1
NN_dir = '{}'.format(pathtoNNmodels)+'NN_{}'.format(NN_filename)    
    
n_summ = theta_fiducial.shape[0]

#loading best model for selected settings of the architecture


files = [int(float(file[-9])) for file in glob.glob('{}/*_err.npy'.format(NN_dir))]
n_members = np.max(files)

if n_members > 1:
    
    training_loss = np.zeros((n_members))
    validation_loss = np.zeros((n_members))
    for i in range (n_members):
        loss = np.load('{}'.format(NN_dir)+'/model{}_loss.npy'.format(i))
        min_index = np.where((loss == np.min(loss[:,1])))[0][0]
        #print(min_index)
        training_loss[i] = loss[min_index,0]
        validation_loss[i] = loss[min_index,1]
        globals()['model_%s'%i] = load_model('{}'.format(NN_dir)+'/model{}_NN.json'.format(i))

    weights_file = Path('{}'.format(NN_dir)+'/stacking_weights.npy')
    if weights_file.is_file():
        print('stacking weights already computed')
        stacking_weights = np.load('{}'.format(weights_file))
    else:
        stacking_weights = np.exp(-np.array([validation_loss[i] for i in range(n_members)]))
        stacking_weights = stacking_weights/sum(stacking_weights)
        np.save('{}'.format(weights_file),stacking_weights)
    
    #compressor
    def compressor(d,compressor_args):    
        comps =  np.zeros((n_members,)+(n_summ,))
        d=np.reshape(d, (n_obs*n_sources,1))
        for i in range(n_members):
            comps[i] = globals()['model_%s'%i].predict(np.moveaxis(d, -1, 0))
        ensemble = np.zeros((n_summ))
        for i in range (comps.shape[0]):
            for j in range (n_summ):
                ensemble[j] += stacking_weights[i]*comps[i,j]
        return ensemble
    
else:
    
    i = 0
    model = load_model('{}'.format(NN_dir)+'/model{}_NN.json'.format(i))
    def compressor(d,compressor_args):
        d=np.reshape(d, (n_obs*n_sources,1))
        x = model.predict(np.moveaxis(d, -1, 0))
        return(x[0])
    


compressor_args=None

if MPI_process == True:
    print('____PROCESS'+ str(rank) + ' for NN_filename=' + NN_filename + 'have been set____')


#quantify NN performance by evaluation of validation data compression residual--------------------------

residual_file = Path('{}/validation_residuals.txt'.format(NN_dir))
if residual_file.is_file():
    print('validation residuals already computed')
else:
    val_data = np.load('../{}'.format(pathtoNNsims)+'/validation_datasets{}.npy'.format(ref_filename))
    for i in range (val_data.shape[0]):    
        val_data[i] = normalization_function(val_data[i])
    val_X = val_data[:,:n_obs*n_sources]
    val_Y = val_data[:,n_obs*n_sources:n_obs*n_sources+2]

    val_pred = np.zeros((val_Y.shape))    
    for i in range (val_pred.shape[0]):
        val_pred[i] = compressor(val_X[i],None)
    residuals = val_Y - val_pred
    plt.hist(residuals)
    plt.close()
    np.savetxt('{}/validation_residuals.txt'.format(NN_dir),residuals) 
    
    
#------------------------------------------------------------------------------

             
# SS vs parameters plots

prerun_theta = np.loadtxt('../{}'.format(pathtopydelfiprerunsims)+'/prerun_thetas{}.txt'.format(ref_filename))
compressed_sims_file = Path('{}'.format(NN_dir)+'/compressed_prerun_sims{}_2000.txt'.format(ref_filename))
if compressed_sims_file.is_file():
    print('prerun sims already compressed')
    compressed_prerun_sims = np.loadtxt('{}'.format(compressed_sims_file))
else:
    prerun_sims = np.loadtxt('../{}'.format(pathtopydelfiprerunsims)+'/prerun_sims{}.txt'.format(ref_filename))
    full_sims = np.column_stack((prerun_sims,prerun_theta))

    for i in range (full_sims.shape[0]):
        full_sims[i] = normalization_function(full_sims[i])

    compressed_prerun_sims = np.zeros((prerun_sims.shape[0],2))
    for i in range(compressed_prerun_sims.shape[0]):
        compressed_prerun_sims[i]=compressor(full_sims[i,:n_obs*n_sources],None) 
    np.savetxt('{}'.format(compressed_sims_file),compressed_prerun_sims)
        

ss_plot_file = Path('{}/SS_vs_parameters.png'.format(NN_dir))
if ss_plot_file.is_file():
    print('SS vs params already plotted')
else:
    fig, axs = plt.subplots(2, 2, figsize=(10,10))
    fig.subplots_adjust(hspace=0.25)
    for i in range (n_params):
        for j in range (2):
            axs[j][i].scatter(full_sims[:,n_obs*n_sources+i],compressed_prerun_sims[:,j],alpha=0.5)
            axs[j][i].scatter(val_Y[:,i],val_pred[:,j],alpha=0.5)
            axs[j][i].set_ylabel('summary_statistics_%s'%j)
            if i==0:
                axs[j][i].set_xlabel('$H_0$')
            else:
                axs[j][i].set_xlabel('$q_0$')         
    fig.savefig('{}/SS_vs_parameters.png'.format(NN_dir))             
    plt.close()        
            
             
#PYDELFI_______________________________________________________________________________________________________

#useful for single NDEs plots__________________________
from matplotlib import cm
import matplotlib.cm as mpcm
from getdist import plots, MCSamples
#______________________________________________________


if int(tf.__version__[0]) < 2:
    sys.path.insert(3, './pydelfi_t1/')
    import ndes as ndes
    import delfi as delfi
    import score as score
    import priors as priors
else:
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    from pydelfi_t2 import delfi
    from pydelfi_t2 import ndes
#______________________________________________________


if os.path.isfile('real_data_element.npy'):
    index = int(np.load('real_data_element.npy'))
else:
    index = 0
    
real_data = np.loadtxt('../{}'.format(pathtorealsamples)+'/multi_randomdata{}.txt'.format(ref_filename))
real_data = normalization_function(real_data, include_theta = False)
compressed_data = compressor(real_data,None)

if using_prerun_sims == False:
    print('need to be written pydelfi simulator')

n_summ = compressed_data.shape[0]
print(compressed_data)
   
#setting the priors of H0 and q0

if q0_prior==False:
    print('PYDELFI: APPLYING UNIFORM PRIOR')
    if int(tf.__version__[0]) < 2:
        prior = priors.Uniform(lower, upper)     
    else:   
        lower = lower.astype(np.float32)
        upper = upper.astype(np.float32)
        prior = tfd.Blockwise([tfd.Uniform(low=lower[i], high=upper[i]) for i in range(lower.shape[0])])
else:
    print('PYDELFI: APPLYING GAUSSIAN PRIOR')
    prior_mean = np.array([70, -0.55])
    if int(tf.__version__[0]) < 2:    
        prior_covariance = np.eye(2)*np.array([20,0.5])**2
        prior = priors.TruncatedGaussian(prior_mean, prior_covariance, lower, upper)
    else:    
        lower = lower.astype(np.float64)
        upper = upper.astype(np.float64)
        prior_covariance = np.eye(lower.shape[0])*np.array([20,0.5])**2
        prior_stddev = np.sqrt(np.diag(prior_covariance))
        prior = tfd.Blockwise([tfd.TruncatedNormal(loc=prior_mean[i], scale=prior_stddev[i],\
                                               low=lower[i], high=upper[i]) for i in range(lower.shape[0])])
        
# Create an ensemble of NDEs  

if int(tf.__version__[0]) < 2:
    NDEs = [ndes.MixtureDensityNetwork(n_parameters=2, n_data=compressed_data.shape[0], n_components=1,\
                                       n_hidden=[30,30], activations=[tf.tanh, tf.tanh], index=0),
            ndes.MixtureDensityNetwork(n_parameters=2, n_data=compressed_data.shape[0], n_components=2, \
                                       n_hidden=[30,30], activations=[tf.tanh, tf.tanh], index=1),
            ndes.MixtureDensityNetwork(n_parameters=2, n_data=compressed_data.shape[0], n_components=3, \
                                       n_hidden=[30,30], activations=[tf.tanh, tf.tanh], index=2),
            ndes.MixtureDensityNetwork(n_parameters=2, n_data=compressed_data.shape[0], n_components=4, \
                                       n_hidden=[30,30], activations=[tf.tanh, tf.tanh], index=3),
            ndes.MixtureDensityNetwork(n_parameters=2, n_data=compressed_data.shape[0], n_components=5, \
                                       n_hidden=[30,30], activations=[tf.tanh, tf.tanh], index=4),
            ndes.ConditionalMaskedAutoregressiveFlow(n_parameters=2, n_data=compressed_data.shape[0], \
                                                     n_hiddens=[50,50], n_mades=5, act_fun=tf.tanh, index=5)]

else:
    NDEs = [ndes.ConditionalMaskedAutoregressiveFlow(
            n_parameters=n_params,
            n_data=compressed_data.shape[0],
            n_mades=n_mds,
            n_hidden=[50,50], 
            activation=tf.keras.activations.tanh,
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1e-5, seed=None),
            all_layers=True)]

    NDEs += [ndes.MixtureDensityNetwork(
            n_parameters=n_params,
            n_data=compressed_data.shape[0], 
            n_components=i+1,
            n_hidden=[30,30], 
            activation=tf.keras.activations.tanh)
        for i in range(n_mds)]
    


#creating the DELFI object

directory_results = '../results_pydelfi/results_pydelfi_{}'.format(NN_filename)
pathlib.Path(directory_results).mkdir(parents=True, exist_ok=True)
if sys.version[0]==3:
    pathlib.Path('{}'.format(directory_results)).mkdir(parents=True, exist_ok=True)
else:
    if not os.path.exists('{}'.format(directory_results)):
        os.makedirs('{}'.format(directory_results))

if int(tf.__version__[0]) < 2:
    DelfiEnsemble = delfi.Delfi(compressed_data, prior, NDEs, 
                            theta_fiducial = theta_fiducial, 
                            param_limits = [lower, upper],
                            param_names = ['H_0','q_0'], 
                            results_dir = "{}/".format(directory_results))
else:
    DelfiEnsemble = delfi.Delfi(compressed_data, prior, NDEs, 
                            theta_fiducial = theta_fiducial, 
                            param_limits = [lower, upper],
                            param_names = ['H_0','q_0'], 
                            results_dir = "{}/".format(directory_results),
                            filename='T2',
                            progress_bar=True,
                            optimiser=tf.keras.optimizers.Adam,
                            optimiser_arguments=None,
                            dtype=tf.float32)

if MPI_process == True:
    print('____PROCESS'+ str(rank) + ' for NN_filename=' + str(data) \
      + 'has set and started pydelfi____')

if using_prerun_sims==True:
   
    #training
    DelfiEnsemble.load_simulations(compressed_prerun_sims, prerun_theta)   
    DelfiEnsemble.train_ndes(epochs = NDE_training_epochs)
    
else:
    
    #SNL
    n_initial = 200
    n_batch = 200
    n_populations = 20

    DelfiEnsemble.sequential_training(simulator, compressor, n_initial, n_batch, n_populations, patience=10,\
                                  plot=True,save_intermediate_posteriors=True, epochs = NDE_training_epochs)

if MPI_process == True:
    cprint('____PROCESS'+ str(rank) + ' for NN_filename=' + str(data) \
      + 'has finished training pydelfi____', 'red', 'on_yellow')


posterior_samples, posterior_weights, log_posterior_values = DelfiEnsemble.emcee_sample()
np.savetxt('{}/final_posterior.txt'.format(directory_results),posterior_samples)
np.savetxt('{}/posterior_weights.txt'.format(directory_results),posterior_weights)

#DelfiEnsemble.triangle_plot(samples=[posterior_samples], weights=[posterior_weights],\
#                            savefig = True, 
#                            filename = '{}/final_posterior'.format(directory_results))



#plotting STAN and pydelfi samples together_____________________________________________________________________________

cm = mpcm.get_cmap('plasma')
colors = [cm(x) for x in np.linspace(0.01, 0.99, 7)]

names = ["H_0","q_0"]
labels = ["H_0","q_0"]

stan_samples = np.load('{}/'.format(pathtoSTANresults)+'multirnd_STAN_samples.npy')

samples_pydelfi = MCSamples(samples=[posterior_samples], names = names, labels = labels,\
                    label='pydelfi samples' +  ' [$H_0$,$q_0$] = [%s ' %round(np.mean(posterior_samples[:,0]),2) + 
                       '$\pm$%s' %round(np.std(posterior_samples[:,0]),2) + ',%s'
                       %round(np.mean(posterior_samples[:,1]),2)
                       + '$\pm$%s' %round(np.std(posterior_samples[:,1]),2) +']')   
samples_stan = MCSamples(samples=[stan_samples], names = names, labels = labels, \
                    label='STAN samples' +  ' [$H_0$,$q_0$] = [%s ' %round(np.mean(stan_samples[:,0]),2) + 
                       '$\pm$%s' %round(np.std(stan_samples[:,0]),2) + ',%s'
                       %round(np.mean(stan_samples[:,1]),2)
                       + '$\pm$%s' %round(np.std(stan_samples[:,1]),2) +']') 

samples_for_plot = [samples_pydelfi,samples_stan]

g = plots.get_subplot_plotter(width_inch = 12)
g.settings.solid_colors = colors
g.settings.alpha_filled_add = 0.6
g.triangle_plot(samples_for_plot, filled=True, normalized=True)
plt.savefig("{}/pydelfi_vs_pystan.png".format(directory_results))
plt.close()

if MPI_process == True:    
    cprint('____PROCESS'+ str(rank) + ' for NN_filename=' + str(data) \
      + 'has finished____', 'yellow', 'on_red')
