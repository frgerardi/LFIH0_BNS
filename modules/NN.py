#!usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import sys

import numpy as np
import itertools
import matplotlib.pyplot as plt
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
import matplotlib.pyplot as plt
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
            print('WARNING: computing normalization')
        norm_sims = data_generation.data_simulator(theta=theta_fiducial, N=100) 
        if verbose == 1:
            print(norm_sims.shape)
        for i in range (norm_sims.shape[0]):
            plt.scatter(norm_sims[i][:100],norm_sims[i][100:200])
        plt.title('Normalization sims')
        plt.show()
        norm_qnts = np.column_stack((np.mean(norm_sims, axis=0), np.std(norm_sims, axis=0)))
        if verbose == 1:
            print(norm_qnts.shape)
        np.savetxt(norm_file,norm_qnts)
        
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


NN_settings = '_2LR0.1_128_128'
training_data = N_draws*n_s

loss_funct = str(loss_function[0])
#print(loss_funct)

NN_filename = loss_funct +'_ns%s'%training_data + NN_settings + '_bs%s'%bs + '_lr%s'%lr + '_l1%s'%l1

if loss_funct == 'MAE':
    loss_expl = 'mean_absolute_error'
else:
    loss_expl = 'mean_squared_error'

n_summ = 2

if MPI_process == True:
    print('____PROCESS'+ str(rank) + ' for [l1,bs,lr]=' + str(data) + 'have been set____')


def model_construction():
    
    model = Sequential()
    
    if l1 == 0:
    
        model = tf.keras.Sequential(
            [tf.keras.Input(shape=(train_X.shape[1],)),
                tf.keras.layers.Dense(128,kernel_initializer=he_ini),
                tf.keras.layers.LeakyReLU(alpha=0.1),
                tf.keras.layers.Dense(128,kernel_initializer=he_ini),
                tf.keras.layers.LeakyReLU(alpha=0.1), 
                tf.keras.layers.Dense(train_Y.shape[1]),
            ])
        
    else:
        
        model = tf.keras.Sequential(
            [tf.keras.Input(shape=(train_X.shape[1],)),
                tf.keras.layers.Dense(128,kernel_initializer=he_ini, kernel_regularizer=tf.keras.regularizers.l1(l1)),
                tf.keras.layers.LeakyReLU(alpha=0.1),
                tf.keras.layers.Dense(128,kernel_initializer=he_ini, kernel_regularizer=tf.keras.regularizers.l1(l1)),
                tf.keras.layers.LeakyReLU(alpha=0.1), 
                tf.keras.layers.Dense(train_Y.shape[1]),
            ])
        
    return model

   
he_ini=tf.keras.initializers.he_normal(seed=None)
n_summ = n_params



def fit_model(train_X, train_Y,i,N_epochs,batchsize,learning_step):
    # define model
    model = model_construction()
    model.compile(loss=loss_expl, optimizer=tf.keras.optimizers.Adam(learning_step),\
                  metrics=[loss_expl])
    #callbacks
    pathlib.Path('{}'.format(NN_arch)).mkdir(parents=True, exist_ok=True)
    callbacks_list = [tf.keras.callbacks.EarlyStopping(patience=50,verbose=1,restore_best_weights=True)]
    #fit model     
    history = model.fit(train_X, train_Y, epochs=N_epochs, validation_data=(val_X, val_Y),\
                        batch_size=batchsize,callbacks=callbacks_list)    
    model.save('{}'.format(NN_arch)+'/model{}_NN.json'.format(i))
    
    loss = np.zeros((len(history.history['loss']),2))
    loss[:,0] = history.history['loss']
    loss[:,1] = history.history['val_loss']
    err = np.zeros((len(history.history[loss_expl]),2))
    err[:,0] = history.history[loss_expl]
    err[:,1] = history.history['val_'+loss_expl]
    np.save('{}'.format(NN_arch)+'/model{}_loss.npy'.format(i),loss)
    np.save('{}'.format(NN_arch)+'/model{}_err.npy'.format(i),err)
    
    fig= plt.figure(figsize=(14,12))
    plt.plot(history.history['loss'][5:])
    plt.plot(history.history['val_loss'][5:])
    plt.title('model loss',fontsize=26)
    plt.ylabel('loss',fontsize=22)
    plt.xlabel('epoch',fontsize=22)
    plt.legend(['train','validation'], loc='upper left')
    #plt.show()
    fig.savefig('{}'.format(NN_arch)+'/model{}_loss_fig.png'.format(i))
    plt.close()
    
    fig= plt.figure(figsize=(14,12))
    plt.plot(history.history[loss_expl][5:])
    plt.plot(history.history['val_'+loss_expl][5:])
    plt.title('model loss',fontsize=26)
    plt.ylabel(loss_funct,fontsize=22)
    plt.xlabel('epoch',fontsize=22)
    plt.legend(['train','validation'], loc='upper left')
    plt.plot(fontsize=24)
    #plt.show()
    fig.savefig('{}'.format(NN_arch)+'/model{}_err_fig.png'.format(i))
    
    return model


train_data = np.load('../{}'.format(pathtoNNsims)+'/training_datasets{}.npy'.format(ref_filename))#[:training_data,:]
np.random.shuffle(train_data)
for i in range (train_data.shape[0]):
    train_data[i] = normalization_function(train_data[i])
train_X = train_data[:,:n_obs*n_sources]
train_Y = train_data[:,n_obs*n_sources:n_obs*n_sources+2]
print(train_X.shape)

val_data = np.load('../{}'.format(pathtoNNsims)+'/validation_datasets{}.npy'.format(ref_filename))#[:int(training_data/10),:]
np.random.shuffle(val_data)
for i in range (val_data.shape[0]):
    val_data[i] = normalization_function(val_data[i])
val_X = val_data[:,:n_obs*n_sources]
val_Y = val_data[:,n_obs*n_sources:n_obs*n_sources+2]
print(val_X.shape)



NN_arch = '{}'.format(pathtoNNmodels)+'/NN_{}'.format(NN_filename)

# fit all models
N_epochs = 10000
batchsize = bs
learning_step = lr
members = [fit_model(train_X, train_Y,i,N_epochs,batchsize,learning_step) for i in range(n_members)]

