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

import numpy         as np
import scipy.special as scs

from .__AbstractLaw            import AbstractLaw
from .NonParametric.__mean     import mean
from .NonParametric.__std      import std
from .NonParametric.__lmoments import lmoments


#############
## Classes ##
#############

class GPD(AbstractLaw):
	"""
	Class to fit a GPD law with covariates, available methods are:
	
	moments  : use empirical estimator of mean and standard deviation to find
	           scale and shape, possibly with least square regression if
	           covariates are given
	lmoments : Use L-Moments estimation, only in stationary context
	lmoments_experimental: Use non-stationary L-Moments with Quantile
	           Regression, experimental and not published, only
	           used to find an initialization of MLE
	bayesian : Bayesian estimation, i.e. the coefficient fitted is the mean of
	           n_mcmc_iteration sample draw from the posterior P(coef_ | Y)
	mle      : Maximum likelihood estimation
	
	WARNING: 
	========
	For this class, f_loc must be always given, because we fit a pareto beyond
	the loc parameter!
	
	Parameters
	==========
	loc   : location parameter, must be pass with f_loc
	scale : scale parameter
	shape : shape parameter
	"""
	__doc__ += AbstractLaw.__doc__
	
	def __init__( self , method = "MLE" ): ##{{{
		"""
		Initialization of Normal law
		
		Parameters
		----------
		method         : string
			Method called to fit parameters
		
		"""
		AbstractLaw.__init__( self , ["loc","scale","shape"] , method )
	##}}}
	
	
	@property
	def loc(self):##{{{
		return self._lhs.values_["loc"]
	##}}}
	
	@property
	def scale(self):##{{{
		return self._lhs.values_["scale"]
	##}}}
	
	@property
	def shape(self):##{{{
		return self._lhs.values_["shape"]
	##}}}
	
	
	def _fit_moments(self):##{{{
		
		self.coef_ = np.zeros(self._rhs.n_features)
		
		Z = self._Y - self.loc.reshape(self._Y.shape)
		
		coef  = np.zeros(self._rhs.n_features)
		il_b  = 0
		il_e  = il_b + self._rhs.s_global[0]
		isc_b = il_e
		isc_e = isc_b + self._rhs.s_global[1]
		ish_b = isc_e
		ish_e = ish_b + self._rhs.s_global[2]
		
		if not self._lhs.is_fixed("scale"):
			X_scale = self._rhs.c_global[1]
			coef[isc_b:isc_e] = std( Z , X_scale , m_Y = 0 , value = False , link = self._rhs.l_global._l_p[1]._l ).reshape(-1)
		
		if not self._lhs.is_fixed("shape"):
			coef[ish_b] = -1e-8
		
		self.coef_ = coef
	##}}}
	
	def _fit_lmoments( self ): ##{{{
		
		self.coef_ = np.zeros(self._rhs.n_features)
		Z = self._Y - self.loc.reshape(self._Y.shape)
		
		coef  = np.zeros(self._rhs.n_features)
		il_b  = 0
		il_e  = il_b + self._rhs.s_global[0]
		isc_b = il_e
		isc_e = isc_b + self._rhs.s_global[1]
		ish_b = isc_e
		ish_e = ish_b + self._rhs.s_global[2]
		
		## L-moments
		lmom = lmoments( Z )
		if not self._lhs.is_fixed("scale") and not self._lhs.is_fixed("shape"):
			itau     = lmom[0] / lmom[1]
			scale_lm = lmom[0] * ( itau - 1 )
			scale_lm = scale_lm if scale_lm > 0 else 1e-8
			shape_lm = 2 - itau
			coef[isc_b] = scale_lm
			coef[ish_b] = shape_lm
		elif not self._lhs.is_fixed("scale"):
			scale = lmom[0] * ( 1 - self.shape )
			scale[ np.logical_not(scale > 0) ] = 1e-8
			coef[isc_b:isc_e] = mean( scale , self._rhs.c_global[1] , value = False , link = self._rhs.l_global._l_p[1]._l )
		elif not self._lhs.is_fixed("shape"):
			Y = self._Y / self.scale.reshape(-1,1)
			lmom = lmoments(Y)
			itau     = lmom[0] / lmom[1]
			coef[ish_b] = 2 - itau
		
		self.coef_ = coef
	##}}}
	
	def _fit_lmoments_experimental( self ): ##{{{
		
		self.coef_ = np.zeros(self._rhs.n_features)
		Z = self._Y - self.loc.reshape(self._Y.shape)
		
		## First step, find lmoments
		try:
			c_Y = np.hstack( [ c for c in self._rhs.c_global if c is not None ] )
		except:
			c_Y = None
		if c_Y is None:
			self._fit_lmoments()
			return
		lmom = lmoments( self._Y , c_Y )
		
		coef  = np.zeros(self._rhs.n_features)
		il_b  = 0
		il_e  = il_b + self._rhs.s_global[0]
		isc_b = il_e
		isc_e = isc_b + self._rhs.s_global[1]
		ish_b = isc_e
		ish_e = ish_b + self._rhs.s_global[2]
		
		if not self._lhs.is_fixed("scale") and not self._lhs.is_fixed("shape"):
			itau  = lmom[:,0] / lmom[:,1]
			scale = lmom[:,0] * ( itau - 1 )
			shape = 2 - itau
			
			coef[isc_b:isc_e] = mean( scale , self._rhs.c_global[1] , link = self._rhs.l_global._l_p[1]._l , value = False )
			coef[ish_b:ish_e] = mean( shape , self._rhs.c_global[2] , link = self._rhs.l_global._l_p[2]._l , value = False )
			
		elif not self._lhs.is_fixed("scale"):
			scale = lmom[:,0].reshape(-1,1) * ( 1 - self.shape.reshape(-1,1) )
			coef[isc_b:isc_e] = mean( scale , self._rhs.c_global[1] , link = self._rhs.l_global._l_p[1]._l , value = False )
		elif not self._lhs.is_fixed("shape"):
			Y     = self._Y / self.scale.reshape(-1,1)
			lmom  = lmoments( Y , self._rhs.c_global[2] )
			shape = 2 - lmom[:,0] / lmom[:,1]
			coef[ish_b:ish_e] = mean( shape , self._rhs.c_global[2] , link = self._rhs.l_global._l_p[2]._l , value = False )
		
		self.coef_ = coef
	##}}}
	
	
	def _special_fit( self ):##{{{
		if self.method == "moments":
			self._fit_moments()
		elif self.method == "lmoments":
			self._fit_lmoments()
		elif self.method == "lmoments-experimental":
			self._fit_lmoments_experimental()
	##}}}
	
	def _init_MLE( self ): ##{{{
		if self._rhs.l_global._special_fit_allowed:
			self._fit_lmoments()
		else:
			self.coef_ = self._rhs.l_global.valid_point( self )
	##}}}
	
	
	def _negloglikelihood( self , coef ): ##{{{
		
		self.coef_ = coef
		
		## Impossible scale
		if not np.all( self.scale > 0 ):
			return np.inf
		
		## Remove exponential case
		shape = self.shape
		zero_shape = ( np.abs(shape) < 1e-10 )
		if np.any(zero_shape):
			shape[zero_shape] = -1e-10
		
		
		##
		loc   = self.loc.reshape(self._Y.shape)
		scale = self.scale.reshape(self._Y.shape)
		shape = self.shape.reshape(self._Y.shape)
		Z = 1. + shape * ( self._Y - loc ) / scale
		
		if not np.all(Z > 0):
			return np.inf
		res = np.sum( np.log( scale ) + np.log(Z) * ( 1 + 1. / shape ) )
		return res
	##}}}
	
	def _gradient_nlll( self , coef ): ##{{{
		
		self.coef_ = coef
		
		## Remove exponential case
		shape = self.shape
		zero_shape = ( np.abs(shape) < 1e-10 )
		if np.any(zero_shape):
			shape[zero_shape] = -1e-10
		
		
		##
		loc   = self.loc.reshape(self._Y.shape)
		scale = self.scale.reshape(self._Y.shape)
		shape = shape.reshape(self._Y.shape)
		
		Z  = ( self._Y - loc ) / scale
		ZZ = 1. + shape * Z
		C  = ( 1. + 1. / shape ) * Z / ZZ
		
		T0 = 1 / scale - C * shape / scale
		T1 = - np.log(ZZ) / shape**2 + C
		
		jac = self._lhs.jacobian_
		p = 0
		if not self._lhs.is_fixed("scale"):
			jac[p,:,:] *= T0
			p += 1
		if not self._lhs.is_fixed("shape"):
			jac[p,:,:] *= T1
		
		return jac.sum( axis = (0,1) )
	##}}}
	

