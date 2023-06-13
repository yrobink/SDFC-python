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

from .__LHS                import LHS
from ..link.__Univariate   import UnivariateLink
from ..link.__Multivariate import MLConstant
from ..link.__Multivariate import MLLinear
from ..link.__Multivariate import MLTensor


###############
## Class(es) ##
###############

class RHS:
	"""
	Class that contains coef_ to fit, covariates and link function. This class
	contains:
	- self.lhs_ : the Left Hand Side part, updated by this class
	- self.c_global : list of covariate
	- self.l_global : global link function
	- self.s_global : length of coef per params of lhs
	
	The parameter self.coef_ is a property, and when it is set the LHS is
	updated accordingly
	
	"""
	def __init__( self , lhs_ : LHS ): ##{{{
		"""
		d
		"""
		self.lhs_     = lhs_
		self.c_global = None
		self.l_global = None
		self.s_global = None
		self._coef_   = None
	##}}}
	
	def build( self , **kwargs ):##{{{
		"""
		Here five kinds of arguments can be passed:
		- c_<param> : covariate of the param,
		- l_<param> : link function of the param,
		- f_<param> : fixed values of the LHS
		- c_global  : list of all covariates, sorted by lhs order
		- l_global  : global link function generated the LHS
		If c_global is set, all arguments (except l_global) are ignored
		"""
		
		## If global covariate and link functions are defined, just set it
		##================================================================
		if kwargs.get("c_global") is not None:
			self.c_global = kwargs["c_global"]
			self.l_global = kwargs.get("l_global")
			return
		
		## Else loop on lhs to find global parameters
		##===========================================
		self.c_global = []
		self.s_global = []
		l_global      = [] ## This list will be passed to MLTensor
		for lhs in self.lhs_.names:
			## Start with covariate
			if kwargs.get("c_{}".format(lhs)) is not None:
				c = kwargs["c_{}".format(lhs)].squeeze()
				if c.ndim == 1: c = c.reshape(-1,1)
				self.c_global.append(c)
				self.s_global.append(1 + c.shape[1])
			## No covariate, two choices : lhs is 1d or fixed
			else:
				self.c_global.append(None)
				if kwargs.get("f_{}".format(lhs)) is not None:
					self.s_global.append(0)
					self.lhs_._fixed[lhs] = True
				else:
					self.s_global.append(1)
			
			## Now the link functions
			if kwargs.get("f_{}".format(lhs)) is not None:
				l_global.append( MLConstant( kwargs["f_{}".format(lhs)] , n_samples = self.lhs_.n_samples ) )
			else:
				l = kwargs.get("l_{}".format(lhs))
				if l is None or issubclass(l.__class__,UnivariateLink):
					l = MLLinear( c = self.c_global[-1] , l = l , n_samples = self.lhs_.n_samples )
				l_global.append(l)
		
		self.l_global = MLTensor( l_global , self.s_global , n_features = np.sum(self.s_global) , n_samples = self.lhs_.n_samples )
	##}}}
	
	## Properties
	## {{{
	
	@property
	def n_features(self):
		return self.l_global.n_features
	
	@property
	def coef_(self):
		return self._coef_
	
	@coef_.setter
	def coef_( self , coef_ ):
		self._coef_         = np.array( [coef_] ).squeeze().reshape(-1)
		self.lhs_.values_   = self.l_global.transform( self.coef_ , self.c_global )
		self.lhs_.jacobian_ = self.l_global.jacobian( self.coef_ , self.c_global )
	
	##}}}
	

