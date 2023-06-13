# -*- coding: utf-8 -*-

## Copyright(c) 2020 Yoann Robin
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

from .__AbstractLaw        import AbstractLaw
from .NonParametric.__mean import mean


###############
## Class(es) ##
###############

class Exponential(AbstractLaw):
	"""
	Class to fit an Exponential law with covariates, available methods are:
	
	moments  : use empirical estimator of mean and standard deviation to find
	           loc and scale, possibly with least square regression if
	           covariates are given
	bayesian : Bayesian estimation, i.e. the coefficient fitted is the mean of
	           n_mcmc_iteration sample draw from the posterior P(coef_ | Y)
	mle      : Maximum likelihood estimation
	
	Parameters
	==========
	scale : scale parameter
	"""
	__doc__ += AbstractLaw.__doc__
	
	def __init__( self , method = "MLE" ): ##{{{
		"""
		Initialization of Exponential law
		
		Parameters
		----------
		method         : string
			Method called to fit parameters
		
		"""
		AbstractLaw.__init__( self , ["scale"] , method )
	##}}}
	
	@property
	def scale(self):##{{{
		return self._lhs.values_["scale"]
	##}}}
	
	
	def _fit_moments(self):##{{{
		
		X_scale = self._rhs.c_global[0]
		self.coef_ = mean( self._Y , X_scale , self._rhs.l_global._l_p[0]._l , False ).squeeze()
	##}}}
	
	def _special_fit( self ):##{{{
		if self.method == "moments":
			self._fit_moments()
	##}}}
	
	def _init_MLE( self ): ##{{{
		if self._rhs.l_global._special_fit_allowed:
			self._fit_moments()
		else:
			self.coef_ = self._rhs.l_global.valid_point( self )
	##}}}
	
	def _negloglikelihood( self , coef ): ##{{{
		self.coef_ = coef
		if not np.all(self.scale > 0):
			return np.Inf
		
		dshape = self._Y.shape
		return np.sum( np.log(self.scale.reshape(dshape)) - self._Y / self.scale.reshape(dshape) )
	##}}}
	
	def _gradient_nlll( self , coef ): ##{{{
		self.coef_ = coef
		
		if not np.all(self.scale > 0):
			return np.zeros_like(self.coef_) + np.nan
		
		scale = self.scale.reshape(self._Y.shape)
		T0 = 1. / scale - self._Y / scale**2
		jac = self._lhs.jacobian_
		jac[0,:,:] *= T0
		
		return jac.sum( axis = (0,1) )
	##}}}


