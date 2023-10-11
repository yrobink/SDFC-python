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
import scipy.special as scs
from .__AbstractLaw import AbstractLaw

from .NonParametric.__mean     import mean
from .NonParametric.__lmoments import lmoments

###############
## Class(es) ##
###############

class GEV(AbstractLaw):
	"""
	Class to fit a GEV law with covariates, available methods are:
	
	moments  : use empirical estimator of mean and standard deviation to find
	           loc and scale, possibly with least square regression if
	           covariates are given
	lmoments : Use L-Moments estimation, only in stationary context
	lmoments_experimental: Use non-stationary L-Moments with Quantile
	           Regression, experimental and not published, only
	           used to find an initialization of MLE
	bayesian : Bayesian estimation, i.e. the coefficient fitted is the mean of
	           n_mcmc_iteration sample draw from the posterior P(coef_ | Y)
	mle      : Maximum likelihood estimation
	
	Parameters
	==========
	loc   : location parameter
	scale : scale parameter
	shape : shape parameter
	
	Warning
	=======
	The shape parameter is the opposite of the shape parameter from scipy:
	GEV ~ scipy.stats.genextreme( loc = loc , scale = scale , c = - shape )
	"""
	__doc__ += AbstractLaw.__doc__
	
	def __init__( self , method = "MLE" ):##{{{
		"""
		Initialization of GEV law
		
		Parameters
		----------
		method         : string
			Method called to fit parameters
		"""
		AbstractLaw.__init__( self , ["loc","scale","shape"] , method )
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
	
	@property
	def shape(self):##{{{
		return self._lhs.values_["shape"]
	##}}}
	
	@property
	def upper_bound(self):##{{{
		return np.where( self.shape < 0 , self.loc - self.scale / self.shape , np.inf )
	##}}}
	
	@property
	def lower_bound(self):##{{{
		return np.where( self.shape < 0 , - np.inf , self.loc - self.scale / self.shape )
	##}}}
	
	## Fit methods
	##============
	
	def _fit_moments(self):##{{{
		
		coefs = np.zeros(self._rhs.n_features)
		m = np.mean(self._Y)
		s = np.sqrt(6) * np.std(self._Y) / np.pi
		
		iloc   = m - 0.57722 * s
		iscale = max( 0.1 , np.log(s) )
		ishape = 1e-8
		
		il_b  = 0
		il_e  = il_b + self._rhs.s_global[0]
		isc_b = il_e
		isc_e = isc_b + self._rhs.s_global[1]
		ish_b = isc_e
		ish_e = ish_b + self._rhs.s_global[2]
		
		## Fit scale
		if not self._lhs.is_fixed("scale"):
			coefs[isc_b] = self._rhs.l_global._l_p[1]._l.inverse(iscale)
		
		## Fit loc
		if not self._lhs.is_fixed("loc"):
			if self._lhs.is_fixed("scale"):
				iloc = m - 0.57722 * np.exp(self.scale)
				coefs[il_b:il_e] = mean( iloc , self._rhs.c_global[0] , value = False , link = self._rhs.l_global._l_p[0]._l )
			else:
				coefs[il_b] = self._rhs.l_global._l_p[1]._l.inverse(iloc)
		
		## Fit shape
		if not self._lhs.is_fixed("shape"):
			coefs[ish_b] = self._rhs.l_global._l_p[2]._l.inverse(ishape)
		
		self.coef_ = coefs
	##}}}
	
	def _fit_lmoments( self ): ##{{{
		
		coefs = np.zeros(self._rhs.n_features)
		
		lmom = lmoments( self._Y )
		
		tau3  = lmom[2] / lmom[1]
		co    = 2. / ( 3. + tau3 ) - np.log(2) / np.log(3)
		kappa = 7.8590 * co + 2.9554 * co**2
		g     = scs.gamma( 1. + kappa )
		
		
		iscale = lmom[1] * kappa / ( (1 - np.power( 2 , - kappa )) * g )
		iloc   = lmom[0] - iscale * (1 - g) / kappa
		ishape = - kappa
		
		il_b  = 0
		il_e  = il_b + self._rhs.s_global[0]
		isc_b = il_e
		isc_e = isc_b + self._rhs.s_global[1]
		ish_b = isc_e
		ish_e = ish_b + self._rhs.s_global[2]
		
		## Fit scale
		if not self._lhs.is_fixed("scale"):
			coefs[isc_b] = self._rhs.l_global._l_p[1]._l.inverse(iscale)
		
		## Fit loc
		if not self._lhs.is_fixed("loc"):
			if self._lhs.is_fixed("scale"):
				iloc = lmom[0] - self.scale.squeeze() * (1 - g) / kappa
				coefs[il_b:il_e] = mean( iloc , self._rhs.c_global[0] , value = False , link = self._rhs.l_global._l_p[0]._l )
			else:
				coefs[il_b] = self._rhs.l_global._l_p[1]._l.inverse(iloc)
		
		## Fit shape
		if not self._lhs.is_fixed("shape"):
			coefs[ish_b] = self._rhs.l_global._l_p[2]._l.inverse(ishape)
		
		self.coef_ = coefs
		
	##}}}
	
	def _fit_lmoments_experimental(self):##{{{
		
		## First step, find lmoments
		try:
			c_Y  = np.ones((self._Y.size,1))
			for c in self._rhs.c_global:
				if c is None: continue
				if c.ndim == 1: c = c.reshape(-1,1)
				for i in range(c.shape[1]):
					c_Y2 = np.hstack( [c_Y,c[:,i].reshape(-1,1)] )
					if np.linalg.matrix_rank(c_Y2) > c_Y.shape[1]:
						c_Y = c_Y2
			if c_Y.shape[1] > 1:
				c_Y = c_Y[:,1:]
			else:
				c_Y = None
		except:
			c_Y = None
		if c_Y is None or c_Y.size == 0:
			self._fit_lmoments()
			return
		
		lmom = lmoments( self._Y , c_Y )
		
		## Find shape
		def uni_shape_solver(tau):
			bl,bu=-1,1
			fct = lambda x : 3 / 2 + tau / 2 - ( 1 - 3**x ) / (1 - 2**x )
			while fct(bl) * fct(bu) > 0:
				bl *= 2
				bu *= 2
			opt = sco.root_scalar( fct , method = "brenth" , bracket = [bl , bu] )
			return opt.root
		shape_solver = np.vectorize(uni_shape_solver)
		tau3 = lmom[:,2] / lmom[:,1]
		try:
			shape = shape_solver(tau3)
		except:
			co    = 2. / ( 3. + tau3 ) - np.log(2) / np.log(3)
			shape = - 7.8590 * co - 2.9554 * co**2
		
		## Find scale
		gshape = scs.gamma( 1 - shape )
		scale = - lmom[:,1] * shape / ( gshape * ( 1 - 2**shape ) )
		
		if not ~(scale.min() > 0):
			idx = ~(scale > 0)
			scale[idx] = 1e-3
		
		## Find loc
		loc = lmom[:,0] - scale * ( gshape - 1 ) / shape
		
		## And now find coefs
		il_b  = 0
		il_e  = il_b + self._rhs.s_global[0]
		isc_b = il_e
		isc_e = isc_b + self._rhs.s_global[1]
		ish_b = isc_e
		ish_e = ish_b + self._rhs.s_global[2]
		coefs = np.array([])
		if not self._lhs.is_fixed("loc"):
			coefs = np.hstack( (coefs,mean( loc , self._rhs.c_global[0] , value = False , link = self._rhs.l_global._l_p[0]._l )) )
		if not self._lhs.is_fixed("scale"):
			coefs = np.hstack( (coefs,mean( scale , self._rhs.c_global[1] , value = False , link = self._rhs.l_global._l_p[1]._l )) )
		if not self._lhs.is_fixed("shape"):
			coefs = np.hstack( (coefs,mean( shape , self._rhs.c_global[2] , value = False , link = self._rhs.l_global._l_p[2]._l )) )
		
		self.coef_ = coefs
	##}}}
	
	def _fit_last_chance(self): ##{{{
		il_b  = 0
		il_e  = il_b + self._rhs.s_global[0]
		isc_b = il_e
		isc_e = isc_b + self._rhs.s_global[1]
		ish_b = isc_e
		ish_e = ish_b + self._rhs.s_global[2]
		coefs = np.zeros(ish_e)
		if not self._lhs.is_fixed("scale"):
			coefs[isc_b] = 0.5
		if not self._lhs.is_fixed("shape"):
			coefs[ish_b] = -0.1
		self.coef_ = coefs
	##}}}
	
	def _special_fit( self ):##{{{
		if self.method == "moments":
			self._fit_moments()
		elif self.method == "lmoments":
			self._fit_lmoments()
		elif self.method == "lmoments-experimental":
			self._fit_lmoments_experimental()
		elif self.method == "last-chance":
			self._fit_last_chance()
	##}}}
	
	def _init_MLE( self ): ##{{{
		if self._rhs.l_global._special_fit_allowed:
			try:
				self._fit_lmoments_experimental()
			except:
				self._fit_last_chance()
		else:
			self.coef_ = self._rhs.l_global.valid_point( self )
	##}}}
	
	
	def _logZafun( self, Z , alpha ):##{{{
		return alpha * np.log( 1. + self.shape * Z )
	##}}}
	
	def _Zafun( self , Z , alpha ):##{{{
		return np.exp( self._logZafun( Z , alpha ) )
	##}}}
	
	def _negloglikelihood( self , coef ): ##{{{
		
		self.coef_ = coef
		
		## Impossible scale
		if not np.all( self.scale > 0 ):
			return np.inf
		
		## Remove exponential case
		zero_shape = ( np.abs(self.shape) < 1e-10 )
		shape = self.shape
		if np.any(zero_shape):
			shape[zero_shape] = 1e-10
		
		dshape = self._Y.shape
		loc    = self.loc.reshape(dshape)
		scale  = self.scale.reshape(dshape)
		shape  = shape.reshape(dshape)
		
		##
		Z = 1 + shape * ( self._Y - loc ) / scale
		
		if not np.all(Z > 0):
			return np.inf
		
		res = np.sum( ( 1. + 1. / shape ) * np.log(Z) + np.power( Z , - 1. / shape ) + np.log(scale) )
		
		
		return res if np.isfinite(res) else np.inf
	##}}}
	
	def _gradient_nlll( self , coef ): ##{{{
		self.coef_ = coef
		
		## Parameters
		dshape = self._Y.shape
		loc   = self.loc.reshape(dshape)
		scale = self.scale.reshape(dshape)
		shape = self.shape.reshape(dshape)
		shc   = 1 + 1 / shape
		Z     = ( self._Y - loc ) / scale
		ZZ    = 1 + shape * Z
		ZZi   = np.power( ZZ ,  - 1 / shape )
		ZZim1 = np.power( ZZ ,  - shc )
		
		## Compute gradient
		T0 = ZZim1 / scale - shc * shape / ( ZZ * scale )
		T1 = 1 / scale + ZZim1 * Z / scale - shc * shape * Z / ( ZZ * scale )
		T2 = np.log(ZZ) * ZZi / shape**2 - ZZim1 * Z / shape - np.log(ZZ) / shape**2 + shc * Z / ZZ
		
		for T in [T0,T1,T2]:
			if not np.isfinite(T).all():
				return np.zeros_like(self.coef_) + np.nan
		
		
		jac = self._lhs.jacobian_
		p = 0
		if not self._lhs.is_fixed("loc"):
			jac[p,:,:] *= T0
			p += 1
		if not self._lhs.is_fixed("scale"):
			jac[p,:,:] *= T1
			p += 1
		if not self._lhs.is_fixed("shape"):
			jac[p,:,:] *= T2
		
		return jac.sum( axis = (0,1) )
	##}}}
	



