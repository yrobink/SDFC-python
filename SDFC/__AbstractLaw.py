# -*- coding: utf-8 -*-

## Copyright(c) 2020 / 2023 Yoann Robin
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

import warnings

import numpy as np
import scipy.optimize as sco
import scipy.stats as sc

from .core.__LHS import LHS
from .core.__RHS import RHS


###############
## Class(es) ##
###############

class AbstractLaw:
	##{{{
	"""
	
	Attributes
	==========
	<param> : np.array
		Value of param fitted, can be loc, scale, name of param of law, etc.
	method : string
		method used to fit
	coef_  : numpy.ndarray
		Coefficients fitted
	info_  : AbstractLaw.Info
		Class containing info of the fit
	
	
	Fit method
	==========
	
	The method <law>.fit is generic, and takes arguments of the form
	<type for param>_<name of param>, see below.
	In case of Bayesian fit, some others optional parameters are available.
	
	Arguments
	---------
	Y         : numpy.ndarray
		Data to fit
	c_<param> : numpy.ndarray or None
		Covariate of a param to fit
	f_<param> : numpy.ndarray or None
		Fix value of a param
	l_<param> : SDFC.tools.LinkFct (optional)
		Link function of a param
	c_global  : list
		List of covariates, sorted by parameters. *_<param> are ignored if set
	l_global  : Inherit of SDFC.link.MultivariateLink
		Global link function. *_<param> are ignored if set
	
	
	Fit with bootstrap
	==================
	The method <law>.fit_bootstrap takes the same arguments that <law>.fit, with
	two new parameters:
	
	n_bootstrap : int
		Number of resampling
	
	
	Possible attributes of <law>.info_
	==================================
	<law>.info_.cov_         : numpy.array
		Covariance matrix, if MLE is used
	<law>.info_.mle_optim_result : scipy.optimize.OptimizeResult
		Result of the optimization of the MLE, if MLE is used
	<law>.info_.n_bootstrap  : int
		Number of resampling
	<law>.info_.coefs_bs_    : numpy.array[shape = (n_bootstrap,n_features)
		Coef fitted with the bootstrap
	
	
	Optional arguments for MLE fit
	==============================
	init : numpy.array
		Init value, optional
	mle_n_restart : integer
		The MLE needs to find a starting point before optimization, this
		parameter control how many time the fit is restarted with a new random
		starting point
	
	
	Optional arguments for Bayesian fit
	===================================
	prior : None or law or prior
		Prior for Bayesian fit, if None a Multivariate Normal law assuming
		independence between parameters is used, if you set it, this must be a
		class which implement the method logpdf(coef), returning the log of
		probability density function
	mcmc_init: None or vector of initial parameters
		Starting point of the MCMC algorithm. If None, prior.rvs() is called.
	transition: None or function
		Transition function for MCMC algorithm, if None is given a normal law
		N(0,0.1) is used.
	n_mcmc_drawn : None or integer
		Number of drawn for MCMC algorithm, if None, the value 10000 is used.
	
	
	Examples
	========
	Example with a Normal law:
	>>> _,X_loc,X_scale,_ = SDFC.Dataset.covariates(2500)
	>>> loc   = 1. + 0.8 * X_loc
	>>> scale = 0.08 * X_scale
	>>> 
	>>> Y = numpy.random.normal( loc = loc , scale = scale )
	>>> 
	>>> ## Define the Normal law estimator, with the MLE method
	>>> law = SDFC.Normal( method = "MLE" )
	>>>
	>>> ## Now perform the fit, c_loc is the covariate of loc, and c_scale the
	>>> ## covariate of scale, and we pass a link function to scale
	>>> law.fit( Y , c_loc = X_loc , c_scale = X_scale , l_scale = SDFC.link.Exponential() )
	>>> print(law.coef_)
	>>>
	>>> ## But we can assume that scale is stationary, so no covariates are given:
	>>> law.fit( Y , c_loc = X_loc , l_scale = SDFC.link.Exponential() )
	>>> print(law.coef_)
	>>>
	>>> ## Or the loc can be given, so we need to fit only the scale:
	>>> law.fit( Y , f_loc = loc , c_scale = X_scale )
	>>> print(law.coef_)
	"""
	##}}}
	#####################################################################
	
	class _Info(object):##{{{
		def __init__(self):
			pass
	##}}}
	
	
	## Init function
	##==============
	
	def __init__( self , names : list , method : str ): ##{{{
		"""
		Initialization of AbstractLaw
		
		Parameters
		----------
		names  : list
			List of names of the law, e.g. ["loc","scale"] for Normal law.
		method : string
			Method called to fit parameters
		
		"""
		self._method = method.lower()
		self._lhs    = LHS(names,0)
		self._rhs    = RHS(self._lhs)
		self.info_   = AbstractLaw._Info()
	##}}}
	
	
	## Properties
	##===========
	
	@property
	def method(self):##{{{
		return self._method
	##}}}
	
	@property
	def coef_(self):##{{{
		return self._rhs.coef_
	##}}}
	
	@coef_.setter
	def coef_( self , coef_ ): ##{{{
		self._rhs.coef_ = coef_
	##}}}
	
	@property
	def cov_(self):##{{{
		return self.info_.cov_
	##}}}
	
	
	## Fit functions
	##==============
	
	def _random_valid_point(self):##{{{
		"""
		Try to find a valid point in the neighborhood of self.coef_
		"""
		coef_ = self.coef_.copy()
		cov_  = 0.1 * np.identity(coef_.size)
		
		p_coef = coef_.copy()
		n_it   = 1
		while not (np.isfinite(self._negloglikelihood(p_coef)) and np.all(np.isfinite(self._gradient_nlll(p_coef))) ):
			if n_it % 100 == 0: cov_ *= 2
			p_coef = np.random.multivariate_normal( coef_ , cov_ )
			n_it += 1
		self.coef_ = p_coef
	##}}}
	
	def _fit_MLE( self , **kwargs ): ##{{{
		
		try:
			self.coef_ = np.array(kwargs["init"]).ravel()
		except:
			self._init_MLE()
		coef_ = self.coef_.copy()
		
		is_success = False
		n_test     = 0
		max_test   = kwargs.get( "mle_n_restart" )
		if max_test is None: max_test = 2
		
		while not is_success and n_test < max_test:
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				self.info_.mle_optim_result = sco.minimize( self._negloglikelihood , self.coef_ , jac = self._gradient_nlll , method = "BFGS" )
			self.info_.cov_             = self.info_.mle_optim_result.hess_inv
			self.coef_                  = self.info_.mle_optim_result.x
			is_success                  = self.info_.mle_optim_result.success
			if not is_success:
				self._random_valid_point()
			n_test += 1
		
		if not is_success:
			self.coef_ = coef_
		
	##}}}
	
	def _fit_Bayesian( self , **kwargs ):##{{{
		## Find numbers of features
		##=========================
		n_features = self._rhs.n_features
		
		## Define prior
		##=============
		prior = kwargs.get("prior")
		if prior is None:
			prior = sc.multivariate_normal( mean = np.zeros(n_features) , cov = 10 * np.identity(n_features) )
		
		## Define transition
		##==================
		transition = kwargs.get("transition")
		if transition is None:
			transition = lambda x : x + np.random.normal( scale = np.sqrt(np.diag(prior.cov)) / 5 )
		
		## Define numbers of iterations of MCMC algorithm
		##===============================================
		n_mcmc_drawn = kwargs.get("n_mcmc_drawn")
		if n_mcmc_drawn is None:
			n_mcmc_drawn = 10000
		
		## MCMC algorithm
		##===============
		draw   = np.zeros( (n_mcmc_drawn,n_features) )
		accept = np.zeros( n_mcmc_drawn , dtype = bool )
		
		## Init values
		##============
		init = kwargs.get("mcmc_init")
		if init is None:
			init = prior.rvs()
		
		draw[0,:]     = init
		lll_current   = -self._negloglikelihood(draw[0,:])
		prior_current = prior.logpdf(draw[0,:]).sum()
		p_current     = prior_current + lll_current
		
		for i in range(1,n_mcmc_drawn):
			draw[i,:] = transition(draw[i-1,:])
			
			## Likelihood and probability of new points
			lll_next   = - self._negloglikelihood(draw[i,:])
			prior_next = prior.logpdf(draw[i,:]).sum()
			p_next     = prior_next + lll_next
			
			## Accept or not ?
			p_accept = np.exp( p_next - p_current )
			if np.random.uniform() < p_accept:
				lll_current   = lll_next
				prior_current = prior_next
				p_current     = p_next
				accept[i] = True
			else:
				draw[i,:] = draw[i-1,:]
				accept[i] = False
		
		self.coef_ = np.mean( draw[int(n_mcmc_drawn/2):,:] , axis = 0 )
		
		## Update information
		self.info_.draw         = draw
		self.info_.accept       = accept
		self.info_.n_mcmc_drawn = n_mcmc_drawn
		self.info_.rate_accept  = np.sum(accept) / n_mcmc_drawn
		self.info_._cov         = np.cov(draw.T)
	##}}}
	
	def fit( self , Y , **kwargs ): ##{{{
		"""
		<law>.fit
		=========
		
		See global documentation for parameters
		"""
		
		
		## Add Y
		self._Y = Y.reshape(-1,1)
		
		## Init LHS/RHS
		self._lhs.n_samples = Y.size
		self._rhs.build(**kwargs)
		
		if self._rhs.n_features == 0:
			raise ValueError("All parameters are fixed (n_features == 0), no fit")
		
		self.coef_ = np.zeros(self._rhs.n_features)
		## Now fit
		if self._method not in ["mle","bayesian","bayesian-experimental"] and self._rhs.l_global._special_fit_allowed:
			self._special_fit()
		elif self._method == "mle" :
			self._fit_MLE(**kwargs)
		else:
			self._fit_Bayesian(**kwargs)
		
		
	##}}}
	
	def fit_bootstrap( self , Y , n_bootstrap , **kwargs ):##{{{
		"""
		<law>.fit_bootstrap
		===================
		
		See global documentation for parameters
		"""
		
		atleast2d = lambda x : x.reshape(-1,1) if x.ndim == 1 else x
		
		Y_       = atleast2d(Y)
		n_sample = Y_.size
		idxs     = np.random.choice( n_sample , n_sample * (n_bootstrap + 1) , replace = True ).reshape((n_bootstrap+1,n_sample))
		idxs[0,:] = range(n_sample)
		
		coefs_bs = []
		kwargs_bs = kwargs.copy()
		for i in range(n_bootstrap):
			
			idx = idxs[i,:]
			
			if "l_global" in kwargs:
				kwargs_bs["l_global"] = kwargs["l_global"]
				if kwargs.get("c_global") is not None:
					kwargs_bs["c_global"] = []
					for c in kwargs["c_global"]:
						if c is None:
							kwargs_bs["c_global"].append(None)
						else:
							
							kwargs_bs["c_global"].append( atleast2d(c)[idx,:])
			else:
				for lhs in self._lhs.names:
					if kwargs.get( "c_{}".format(lhs) ) is not None:
						kwargs_bs["c_{}".format(lhs)] = atleast2d(kwargs[f"c_{lhs}"])[idx,:]
					if kwargs.get( "f_{}".format(lhs) ) is not None:
						kwargs_bs["f_{}".format(lhs)] = atleast2d(kwargs[f"f_{lhs}"])[idx,:]
			
			if len(coefs_bs) > 0:
				kwargs_bs["init"] = coefs_bs[0]
			
			self.fit( Y_[idx,:] , **kwargs_bs )
			coefs_bs.append(self.coef_.copy())
		
		self.info_.n_bootstrap  = n_bootstrap
		self.info_.coefs_bs_    = np.array(coefs_bs)
		
		self.fit( Y , **kwargs )
	##}}}

