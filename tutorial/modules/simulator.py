#!/usr/bin/env python
# coding: utf-8

import sys

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import pickle
#import IMNN.IMNN as imnn_module
import jax.numpy as jnp
import jax.random as jrnd
import jax.ops as jo
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

from sklearn.utils import shuffle

sys.path.insert(1, './modules/')
import pydelfi_t1.priors as priors

from smt.sampling_methods import LHS

import datetime

from set_params import *

class simulations():
    
    def __init__(self):
        
        self.n_sources = n_sources
        
        self.selection = selection
        
        self.n_obs = n_obs 
        
        self.ref_filename = ref_filename

        self.input_shape = (self.n_sources*self.n_obs,)

        self.theta_fid = theta_fiducial
        self.n_params = n_params  
        
        self.lower = lower
        self.upper = upper
        
        self.draws_method = draws_method
        
        if self.draws_method == 2:
            self.prior_distr = priors.Uniform(lower, upper)
        
        self.sorting = sorting
        
        self.N_draws = N_draws
        
        self.n_s = n_s
        self.n_s_val = n_s_val
       
        self.m_bins = interpolation_bins
        self.zmin = z_range[0]
        self.zmax = z_range[1]

        self.sigmaz = sigmaz
        self.meanv = meanv
        self.sigmavlos = sigmavlos
        self.sigmav = sigmav

        
        #CONSTANTS
        self.c = c
        self.G = G 
        
               
        
    def redshift_distribution(self,theta):     #fixed

        tot_bins=self.m_bins
        zb=np.linspace((self.zmax/self.m_bins),self.zmax,tot_bins)
        r=4*np.pi*(self.c**3)*(zb**2)*(1-2*zb*(1+theta[1]))/(theta[0]**3)   #dV/dz ratio eq13 of the paper 
        pdf=r/(1+zb)                                                        #probability density function

        cdf=np.zeros(zb.shape[0])
        cdf[0]=pdf[0]    
        for i in range (1,zb.shape[0]):
            cdf[i]=cdf[i-1]+pdf[i]
        
        #now let's focus on the only portion of the CDF we are interested (z_min,z,z_max)
        if self.zmin!=0.:
            range_zb=np.linspace(self.zmin,self.zmax,self.m_bins)
            range_cdf = np.zeros((self.m_bins))
            for i in range (self.m_bins):
                range_cdf[i]=np.interp(range_zb[i], zb, cdf)    
            normcdf=range_cdf/range_cdf[self.m_bins-1]          #normalizing the cdf
        
        else:
            range_zb=zb
            range_cdf=cdf
            normcdf=cdf/cdf[self.m_bins-1]
        
        return np.column_stack((range_zb,normcdf))
    
   

    
    def data_simulator(self, theta, N=1):
          
        sims_output = np.zeros((N,) + (self.n_obs*self.n_sources,))
        data = np.zeros(((self.n_sources,) + (self.n_obs+2,)))
        
        zb_cdf = self.redshift_distribution(theta)
        
        if sorting == 'noisy redshift':  sorting_index = 0
        else:   sorting_index = 3
            
        for i in range (N):
    
            k = 0
            
            while k < n_sources:
                    
                random_cdf = rnd.uniform(zb_cdf[0,1],zb_cdf[self.m_bins-1,1])
                normal_draws = rnd.normal(size=(4))
                redshift = np.interp(random_cdf, zb_cdf[:,1],zb_cdf[:,0])
                D=c*redshift*(1+(1-theta[1])*redshift/2)/theta[0]
                        
                v= self.meanv + self.sigmavlos*normal_draws[0]
                z_tot = redshift + (1+redshift)*v/self.c
                v_noisy = v + self.sigmav*normal_draws[1]
                z_noisy = z_tot + self.sigmaz*normal_draws[2]
        
                D_noisy = D + 0.1*D*normal_draws[3]
                rho = 12 * 250/D_noisy
                         
                if (selection == False) or (selection == True and rho>12): 
                    
                    data[k] = np.array([z_noisy, D_noisy, v_noisy, redshift, rho])                                        
                    k=k+1                       

            data = data[data[:,sorting_index].argsort()]
                
            sims_output[i,:] = jnp.concatenate((data[:,0],data[:,1],data[:,2]))
                    
                
        return sims_output
    

    def NN_datasets(self,save_datasets=True):
        
        if sys.version[0]==3:
            pathlib.Path('{}'.format(pathtoNNsims)).mkdir(parents=True, exist_ok=True) 
        else:
            if not os.path.exists('{}'.format(pathtoNNsims)):
                os.makedirs('{}'.format(pathtoNNsims))     
        
        if self.draws_method == 1:
            theta_limits = np.array([[self.lower[0],self.upper[0]],[self.lower[1],self.upper[1]]])
            sampling = LHS(xlimits=theta_limits)
            theta_draws = sampling(self.N_draws)
        else:   
            theta_draws = np.zeros((N_draws,2))
            for i in range(N_draws):
                theta_draws[i] = self.prior_distr.draw()
                
                
        for m in range (2):
            
            if m == 0:   #training set
                N = n_s
                output_file = 'training_datasets'+self.ref_filename
                print('computing training simulations')
            else:        #validation set
                N = n_s_val
                output_file = 'validation_datasets'+self.ref_filename
                print('computing validation simulations')
                
            output = np.zeros((N,) + (theta_draws.shape[0],) + (self.n_obs*self.n_sources+2,))
            print(output.shape)
            
            t0 = datetime.datetime.now().replace(microsecond=0)
        
            for a in range (theta_draws.shape[0]):
                              
                if a% 100 == 0: print('%s'%a + '/%s'%theta_draws.shape[0]  + ' thetas done')
            
                theta = theta_draws[a]
                
                output[:,a,:self.n_obs*self.n_sources] = self.data_simulator(theta,N)  
                #print(output.shape)
                for f in range (2):
                    output[:,a,self.n_obs*self.n_sources+f] = theta[f] 
                    
                
                if a% 100 == 0: 
                    t1 = datetime.datetime.now().replace(microsecond=0)
                    print('time for all sims', t1-t0)   
            
            prep_output = np.zeros((output.shape[0]*output.shape[1],)+(self.n_obs*self.n_sources+2,))     
            for a in range (theta_draws.shape[0]):
                for i in range (N):
                    prep_output[i+a*N,:] = output[i,a,:]
                  
            np.save('{}'.format(pathtoNNsims)+'/{}'.format(output_file),prep_output)
            

    def data_simulator_for_N(self, theta, N):

        output_array = np.zeros(((N,5)))

        np.random.seed(647392)
            
        zb_cdf=self.redshift_distribution(theta)
        zb=zb_cdf[:,0]
        cdf=zb_cdf[:,1] 
        
        for l in range (N):   
            
            random_cdf=np.random.uniform(cdf[0],cdf[self.m_bins-1])
            z=np.interp(random_cdf, cdf, zb)    
            D=self.c*z*(1+(1-theta[1])*z/2)/theta[0]
            
            v=np.random.normal(self.meanv, self.sigmavlos)
            z_tot = z + (1+z)*v/self.c
            v_noisy = np.random.normal(v, self.sigmav)
            z_noisy = np.random.normal(z_tot, self.sigmaz)
            
            sigma_D = 0.1*D
            D_noisy = D + np.random.normal(0, sigma_D)
            rho = 12 * 250/D_noisy
                
            output_array[l,0]=z_noisy
            output_array[l,1]=D_noisy
            output_array[l,2]=rho
            output_array[l,3]=D
            output_array[l,4]=sigma_D
               
               
        return output_array
