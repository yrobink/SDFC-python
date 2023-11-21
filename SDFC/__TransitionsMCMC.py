# -*- coding: utf-8 -*-

## Copyright(c) 2020 / 2023 Occitane Barbaux and Yoann Robin
## 
## This file is part of SDFC.
## 
## SDFC is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
## 
## SDFC is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with SDFC.  If not, see <https://www.gnu.org/licenses/>.


##############
## Packages ##
##############
import numpy as np


##############
## Functions##
##############

def transition_fixed(x,tran_scale_G):

    """
    Fixed Transition Function For Metropolis-Hasting
    Multivariate normal centered on 0
    tran_scale_G give the covariance 
    Ok for 1 or 5 dimensions
    
    Parameters
    ----------
    x  : np.array
        One or All dimensions, past value
    tran_scale_G: np.array
        Covariance, one or all dimensions  
    
    Outputs
    ----------
    x1  : float
        New values sampled (same dimension as x )

    
    
    """
    if len(x.shape)==0:
    	length_vect=1
    	x1=x+np.random.normal(scale=tran_scale_G)
    else:
    	length_vect=x.shape[0]
    	
    	if len(tran_scale_G.shape)==1:
    	#(can be a 1d vector or a unique value)
        	tran_scale_G=np.identity(length_vect)*(tran_scale_G*tran_scale_G)
    #(or a 5d matrix)
    	x1=x+np.random.default_rng().multivariate_normal( mean=[0]*length_vect,cov = tran_scale_G )
    #Used to be np.random.normal(scale=tran_scale_G)
    return(x1)

#Adaptative MH
def transition_adaptative(x,i,draw,init=0.01,epsilon=0.01):

    """
    Adaptative Transition Function For Metropolis-Hasting
    Adaptative Metropolis (Haario et al. 2001), Based on ([Craiu et Rosenthal, 2014, p. 189] 
    Multivariate normal centered on 0
    Use past draw to calculate covariance 
    All dimension simultaneously
    
    Parameters
    ----------
    x  : np.array
        All dimensions, past value
    i : int
        which dimension is treated
    draw :np.array
    	Past samples for cov calculation
    initTrans : float
        Variance during the pre-period used to then start the adaptation.   
    epsilon  : float
        Noise added so the variance do not goes to 0.  
    
    Outputs
    ----------
    x1  : float
        New values sampled 

    
    
    """
    print("Adapt")
    if i<500:
        sigma=np.identity(x.shape[0])*(init)
        #pre period
    else:
        #Adaptative period
        sigma=np.cov(draw,rowvar=False)*(2.38*2.38/draw.shape[1])+np.identity(x.shape[0])*(epsilon)
    
    
    x1=x+np.random.default_rng().multivariate_normal( mean=[0]*x.shape[0],cov = sigma )
    #Used to be np.random.normal(scale=tran_scale_G)
    return(x1)



def transition_SCAM(x ,i,draw,initTrans=0.01,epsilon=0.01):
    """
    Adaptative Transition Function For Metropolis-Hasting Within Gibbs
    Adaptative Metropolis (Haario et al. 2005), Based on ([Roberts et Rosenthal, 2009,]) 
    Univariate normal centered on 0
    Use past draw to calculate covariance 
    1 dimension
    
    Parameters
    ----------
    x  : float
        Past value
    i : int
        iteration number
    prev_sigma  : *list
        Not Used Yet ! For faster calculation of adaptative sigma
    initTrans : float
        Variance during the pre-period used to then start the adaptation.   
    epsilon  : float
        Noise added so the variance do not goes to 0.  
    
    Outputs
    ----------
    x1  : float
        New value sampled for the dimension of interest
    
    
    """
    if i<500:
        sigma=initTrans
        #pre period
    elif i==500:
        sigma=np.var(draw)*2.4+epsilon
        #start adaptation
    else:
        #Adaptative period
        
        #could be faster, to be added
        #x_bar=np.mean(draw)
        #g_prev=(prev_sigma-0.01)/2.4
        #gt=(i-1)/i*g_prev+np.mean(draw[:(i-1)])**2+x**2/i-(i+1)/i*np.mean(draw)**2
        
        gt=np.var(draw)
        sigma=(2.4**2)*(gt+epsilon)

    
    x1=x+np.random.normal( size = 1 , scale = sigma )

    return(x1)

