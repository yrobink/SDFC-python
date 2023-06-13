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
from .__AbstractLaw import AbstractLaw

from .NonParametric.__mean import mean
from .NonParametric.__std  import std


###############
## Class(es) ##
###############

class Normal(AbstractLaw):
	"""
	Class to fit a Normal law with covariates, available methods are:
	
	moments  : use empirical estimator of mean and standard deviation to find
	           loc and scale, possibly with least square regression if
	           covariates are given
	bayesian : Bayesian estimation, i.e. the coefficient fitted is the mean of
	           n_mcmc_iteration sample draw from the posterior P(coef_ | Y)
	mle      : Maximum likelihood estimation
	
	Parameters
	==========
	loc   : location parameter
	scale : scale parameter
	"""
	__doc__ += AbstractLaw.__doc__
	
	def __init__( self , method = "MLE" ):##{{{
		"""
		Initialization of Normal law
		
		Parameters
		----------
		method : string
			Method called to fit parameters
		
		"""
		AbstractLaw.__init__( self , ["loc","scale"] , method )
		self._loc   = None
		self._scale = None
	##}}}
	
	## Properties
	##===========
	
	@property
	def loc(self):##{{{
		return self._lhs.values_["loc"]
	##}}}
	
	@property
	def scale(self):##{{{
		return self._lhs.values_["scale"]
	##}}}
	
	
	## Fit methods
	##============
	
	def _fit_moments( self ): ##{{{
		
		coefs = np.zeros(np.sum(self._rhs.l_global._s_p))
		
		## Find loc
		##=========
		if not self._lhs.is_fixed("loc"):
			X_loc = self._rhs.c_global[0]
			coefs[:self._rhs.l_global._s_p[0]] = mean( self._Y , X_loc , self._rhs.l_global._l_p[0]._l , False ).squeeze()
			self.coef_ = coefs
		
		## Find scale
		##===========
		if not self._lhs.is_fixed("scale"):
			X_scale = self._rhs.c_global[1]
			coefs[self._rhs.l_global._s_p[0]:] = std( self._Y , X_scale , self.loc , self._rhs.l_global._l_p[1]._l , False ).squeeze()
			self.coef_ = coefs
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
		shape = self._Y.shape
		scale2 = np.power( self.scale , 2 )
		if not np.isscalar(scale2): scale2 = scale2.reshape(shape)
		return np.Inf if not np.all( self.scale > 0 ) else np.sum( np.log( scale2 ) ) / 2. + np.sum( np.power( self._Y - self.loc.reshape(shape) , 2 ) / scale2 ) / 2.
	##}}}
	
	def _gradient_nlll( self , coef ): ##{{{
		self.coef_ = coef
		## Parameters
		shape = self._Y.shape
		loc   = self.loc.reshape(shape)
		scale = self.scale.reshape(shape)
		Z     = ( self._Y - loc ) / scale
		
		## Compute gradient
		T0 = - Z / scale
		T1 = - self._Y * Z / scale**2 + loc * Z / scale**2 + 1 / scale
		jac = self._lhs.jacobian_
		p = 0
		if not self._lhs.is_fixed("loc"):
			jac[p,:,:] *= T0
			p += 1
		if not self._lhs.is_fixed("scale"):
			jac[p,:,:] *= T1
		
		return jac.sum( axis = (0,1) )
	##}}}
	

