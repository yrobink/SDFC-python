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


###############
## Class(es) ##
###############


class UnivariateLink:##{{{
	"""
	SDFC.link.UnivariateLink
	========================
	base class for univariate link
	"""
	def __init__(self): pass
	
	def __call__( self , x ):
		return self.transform(x)
##}}}

class ULIdentity(UnivariateLink): ##{{{
	"""
	SDFC.link.ULIdentity
	====================
	
	Identity link function, i.e.:
		f(x) = x
		f^{-1}(x) = x
		df(x) = 1
	"""
	
	def __init__( self ):
		UnivariateLink.__init__(self)
	
	def transform( self , x ):
		return x
	
	def inverse( self , x ):
		return x
	
	def jacobian( self , x ):
		return np.ones_like(x)
##}}}

class ULExponential(UnivariateLink): ##{{{
	"""
	SDFC.link.ULExponential
	=======================
	
	Exponential link function, i.e.:
		f(x) = exp(s*x) + b
		f^{-1}(x) = log(x-b) / s
		df(x) = s*exp(s*x)
	This function is used to bound a variable by the level b, by upper if s > 0 or lower if s < 0.
	"""
	
	def __init__( self , b = 0 , s = 1 ):
		UnivariateLink.__init__(self)
		self.b = b
		self.s = s
	
	def transform( self , x ):
		return np.exp( self.s * x) + self.b
	
	def inverse( self , x ):
		return np.log(x - self.b) / self.s
	
	def jacobian( self , x ):
		return self.s * np.exp(self.s * x)
##}}}

class ULInverse(UnivariateLink): ##{{{
	"""
	SDFC.link.ULInverse
	===================
	
	Inverse link function, i.e.:
		f(x) = 1/x
		f^{-1}(x) = 1/x
		df(x) = - 1 / x**2
	"""
	def __init__(self):
		pass
	
	def transform( self , x ):
		return 1. / x
	
	def inverse( self , x ):
		return 1. / x
	
	def jacobian( self , x ):
		return - 1. / x**2
##}}}

class ULLogit(UnivariateLink):##{{{
	"""
	SDFC.link.ULLogit
	=================
	
	Logit link function, i.e.:
		f(x) = ( b - a ) / ( 1 + exp(-sx) ) + b
		f^{-1}(x) = - log( (b-a) / (x-b) - 1) / s
		df(x) = s * (b-a) * exp( -sx ) / ( 1 + exp(-sx) )**2
	
	This function constrain the parameters estimated between a and b, and the parameters s control the growth between a and b.
	"""
	def __init__( self , a = 0 , b = 1 , s = 1 ):
		self.a      = a
		self.b      = b
		self.s      = s
	
	def transform( self , x ):
		return (self.b - self.a) / ( 1. + np.exp( - x * self.s ) ) + self.a
	
	def inverse( self , x ):
		x = np.array( [x] ).ravel()
		idx_lo = x < self.a
		idx_up = x > self.b
		x[idx_lo] = self.a + 1e-3
		x[idx_up] = self.b - 1e-3
		return - np.log( (self.b - self.a) / ( x - self.a ) - 1 ) / self.s
	
	def jacobian( self , x ):
		e = np.exp( - self.s * x )
		return self.s * (self.b - self.a) * e / ( 1 + e )**2
##}}}

class ULCustom(UnivariateLink): ##{{{
	"""
	SDFC.link.ULCustom
	==================
	
	Class to define a custom link function for users.
	
	Three functions must be given:
	- tranform : the function to apply the link function
	- inverse  : the inverse of the transform
	- jacobian : the derivative of the transform
	
	Example
	-------
	
	>>> ## The exponential link function is equivalent to:
	>>> custom = SDFC.link.ULCustom( np.exp , np.log , np.exp )
	
	"""
	
	def __init__( self , transform , inverse , jacobian ):
		UnivariateLink.__init__(self)
		self._transform = transform
		self._inverse   = inverse
		self._jacobian  = jacobian
	
	def transform( self , x ):
		return self._transform(x)
	
	def inverse( self , x ):
		return self._inverse(x)
	
	def jacobian( self , x ):
		return self._jacobian(x)
##}}}

