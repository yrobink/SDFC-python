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
from .__Univariate import ULIdentity


###############
## Class(es) ##
###############


class MultivariateLink:##{{{
	"""
	SDFC.link.MultivariateLink
	==========================
	Base class for MultivariateLink
	
	"""
	
	def __init__( self , *args , **kwargs ):##{{{
		self._special_fit_allowed = False
		self._n_features = kwargs.get("n_features")
		self._n_samples  = kwargs.get("n_samples")
	##}}}
	
	def transform( self , coef , X ):##{{{
		pass
	##}}}
	
	def jacobian( self , coef , X ):##{{{
		pass
	##}}}
	
	@property
	def n_features(self):##{{{
		return self._n_features
	##}}}
	
	@property##{{{
	def n_samples(self):
		return self._n_samples
	##}}}
	
##}}}

class MLConstant(MultivariateLink):##{{{
	"""
	SDFC.link.MLConstant
	====================
	Link used for fixed parameters f_<param>
	
	"""
	
	def __init__( self , value , *args , **kwargs ):##{{{
		kwargs["n_features"] = 0
		MultivariateLink.__init__( self , *args , **kwargs )
		self.value_ = np.array([value]).reshape(-1)
		if self.value_.size == 1:
			self.value_ = np.repeat( value , self.n_samples )
	##}}}
	
	def transform( self , *args , **kwargs ):##{{{
		return self.value_
	##}}}
	
##}}}

class MLLinear(MultivariateLink): ##{{{
	"""
	SDFC.link.MLLinear
	==================
	Link function which contains an univariate link function. The idea is to
	chain a linear map with an univariate transform
	
	"""
	
	def __init__( self , *args , **kwargs ):##{{{
		self._l     = kwargs.get("l")
		self._c     = kwargs.get("c")
		if self._l is None: self._l = ULIdentity()
		if self._c is not None: kwargs["n_samples"] = self._c.shape[0]
		MultivariateLink.__init__( self , *args , **kwargs )
		
		self.design_ = np.ones( (self.n_samples,1) )
		if self._c is not None:
			self.design_ = np.hstack( (self.design_,self._c) )
		
	##}}}
	
	def _linear_transform( self , coef , X ): ##{{{
		return self.design_ @ coef
	##}}}
	
	def transform( self , coef , X ):##{{{
		out = self._l.transform( self._linear_transform(coef,X) )
		return out
	##}}}
	
	def jacobian( self , coef , X ): ##{{{
#		jac = np.zeros( (self.n_samples,self.n_features) )
#		jac[:,0]  = 1
#		jac[:,1:] = X
		return self._l.jacobian( self._linear_transform( coef , X ).reshape(-1,1) ) * self.design_
	##}}}
	
##}}}

class MLTensor(MultivariateLink):##{{{
	"""
	SDFC.link.MLTensor
	==================
	Link function used to build the product of univariate link function
	
	
	"""
	
	def __init__( self , l_p , s_p , *args , **kwargs ):##{{{
		kwargs["n_features"] = np.sum(s_p)
		MultivariateLink.__init__( self , *args , **kwargs )
		self._l_p = l_p
		self._s_p = s_p
		self._special_fit_allowed = np.all( [isinstance(l,(MLLinear,MLConstant)) for l in self._l_p] )
	##}}}
	
	def transform( self , coef , X ): ##{{{
		list_p = []
		ib,ie = 0,0
		for s,l,x in zip(self._s_p,self._l_p,X):
			ie += s
			list_p.append( l.transform( coef[ib:ie] , x ) )
			ib += s
		return list_p
	##}}}
	
	def jacobian( self , coef , X ): ##{{{
		list_jac = []
		ib,ie = 0,0
		jac = np.zeros( (np.nonzero(self._s_p)[0].size,self.n_samples,self.n_features) )
		i = 0
		for s,l,x in zip(self._s_p,self._l_p,X):
			if s > 0:
				ie += s
				jac[i,:,ib:ie] = l.jacobian( coef[ib:ie] , x )
				ib += s
				i += 1
		return jac
	##}}}
	
##}}}



