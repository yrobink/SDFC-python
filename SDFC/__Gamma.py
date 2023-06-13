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

import numpy          as np
import scipy.special  as scs

from .__AbstractLaw        import AbstractLaw
from .NonParametric.__mean import mean
from .NonParametric.__var  import var


#############
## Classes ##
#############

class Gamma(AbstractLaw):
	"""
	Class to fit a Gamma law with covariates, available methods are:
	
	moments  : use empirical estimator of mean and standard deviation to find
	           scale and shape, possibly with least square regression if
	           covariates are given
	bayesian : Bayesian estimation, i.e. the coefficient fitted is the mean of
	           n_mcmc_iteration sample draw from the posterior P(coef_ | Y)
	mle      : Maximum likelihood estimation
	
	Parameters
	==========
	scale : scale parameter
	shape : shape parameter
	"""
	__doc__ += AbstractLaw.__doc__
	
	def __init__( self , method = "MLE" ): ##{{{
		"""
		Initialization of Gamma law
		
		Parameters
		----------
		method         : string
			Method called to fit parameters
		"""
		AbstractLaw.__init__( self , ["scale","shape"] , method )
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
		
		n_samples = self._lhs.n_samples
		
		self.coef_ = np.zeros(self._rhs.n_features)
		
		if not self._lhs.is_fixed("scale") and not self._lhs.is_fixed("shape"):
			mX = np.ones( (n_samples,1) )
			vX = np.ones( (n_samples,1) )
			design_scale = self._rhs.l_global._l_p[0].design_
			design_shape = self._rhs.l_global._l_p[1].design_
			for i in range(1,design_scale.shape[1]):
				for j in range(design_shape.shape[1]):
					mX = np.hstack( (mX,np.reshape( design_scale[:,i]    * design_shape[:,j] , (n_samples,1) ) ) )
					vX = np.hstack( (vX,np.reshape( design_scale[:,i]**2 * design_shape[:,j] , (n_samples,1) ) ) )
			m = mean( self._Y , mX[:,1:] )
			v = var(  self._Y , vX[:,1:] )
			
			
			idx  = np.logical_or( np.abs(m) < 1e-8 , v < 1e-8 )
			cidx = np.logical_not(idx)
			scale = np.zeros_like(m)
			shape = np.zeros_like(m)
			scale[cidx] = np.abs( v[cidx] / m[cidx] )
			shape[cidx] = np.abs( m[cidx]**2 / v[cidx] )
			
			if np.any(idx):
				scale[idx] = scale[cidx].min()
				shape[idx] = shape[cidx].min()
			
			coef = np.array([])
			coef = np.hstack( (coef,mean( scale , design_scale[:,1:] , value = False , link = self._rhs.l_global._l_p[0]._l ).reshape(-1)) )
			coef = np.hstack( (coef,mean( shape , design_shape[:,1:] , value = False , link = self._rhs.l_global._l_p[1]._l ).reshape(-1)) )
			
		elif self._lhs.is_fixed("scale"):
			
			
			design_shape = self._rhs.l_global._l_p[1].design_
			m = mean( self._Y  , design_shape[:,1:] * self.scale.reshape(-1,1) )
			v = var(  self._Y  , design_shape[:,1:] * self.scale.reshape(-1,1)**2 )
			
			shape = np.abs( m**2 / v )
			coef = mean( shape , design_shape[:,1:] , value = False , link = self._rhs.l_global._l_p[1]._l )
			
		elif self._lhs.is_fixed("shape"):
			design_scale = self._rhs.l_global._l_p[0].design_
			m = mean( self._Y  , design_scale[:,1:]    * self.shape.reshape(-1,1) )
			v = var(  self._Y  , design_scale[:,1:]**2 * self.shape.reshape(-1,1) )
			
			scale = np.abs( v / m )
			coef = mean( scale , design_scale[:,1:] , value = False , link = self._rhs.l_global._l_p[0]._l )
		
		self.coef_ = coef
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
		if not np.all(self.scale > 0) or not np.all(self.shape > 0) or not np.all(self._Y > 0):
			return np.Inf
		
		scale = self.scale.reshape(self._Y.shape)
		shape = self.shape.reshape(self._Y.shape)
		
		return np.sum( self._Y / scale + scs.loggamma(shape) + shape * np.log(scale) - (shape-1) * np.log(self._Y) )
	##}}}
	
	def _gradient_nlll( self , coef ): ##{{{
		
		self.coef_ = coef
		
		if not (np.all(self.scale > 0) and np.all(self.shape > 0) and np.all(self._Y > 0)):
			return np.zeros(self.coef_.size) + np.nan
		
		dshape = self._Y.shape
		scale = self.scale.reshape(dshape)
		shape = self.shape.reshape(dshape)
		
		## Gradient
		T0 = - self._Y / scale**2 + shape / scale
		T1 = scs.digamma(shape) + np.log(scale) - np.log(self._Y)
		
		## Compute jacobian
		jac = self._lhs.jacobian_
		p = 0
		if not self._lhs.is_fixed("scale"):
			jac[p,:,:] *= T0
			p += 1
		if not self._lhs.is_fixed("shape"):
			jac[p,:,:] *= T1
		
		return jac.sum( axis = (0,1) )
	##}}}
	




