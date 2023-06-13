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
import scipy.linalg as scl

from .__Multivariate import MultivariateLink


###############
## Class(es) ##
###############

class NormalRatioLocScaleConstant(MultivariateLink):
	"""
	SDFC.link.NormalRatioLocScaleConstant
	=====================================
	Global link function for Normal law with three rhs parameter, giving:
	
	loc   = loc0   * exp( alpha / loc0 * X )
	scale = scale0 * exp( alpha / loc0 * X )
	
	The vector (loc0,scale0,alpha) is fitted with this link function.
	
	"""
	
	def __init__( self , n_samples ):##{{{
		MultivariateLink.__init__( self , n_features = 3 , n_samples = n_samples )
		self._l_p = [None,None]
	##}}}
	
	def transform( self , coef , X ):##{{{
		XX = X[0] if type(X) == list else X
		E = np.exp( coef[2] / coef[0] * XX[:,0] )
		loc   = coef[0] * E
		scale = coef[1] * E
		return loc,scale
	##}}}
	
	def jacobian( self , coef , X ):##{{{
		XX = X[0] if type(X) == list else X
		E = np.exp( coef[2] / coef[0] * XX[:,0] )
		jac = np.zeros( (2 ,  self.n_samples , self.n_features ) )
		jac[0,:,0] = E - coef[2] * XX[:,0] / coef[0] * E
		jac[1,:,0] = - coef[1] * coef[2] * XX[:,0] / coef[0]**2 * E
		jac[1,:,1] = E
		jac[0,:,2] = XX[:,0] * E
		jac[1,:,2] = coef[1] * XX[:,0] * E / coef[0]
		
		return jac
	##}}}
	
	def valid_point( self , law ):##{{{
		
		## Fit by assuming linear case without link functions
		linear_law = type(law)()
		l_c = [ c for c in law._rhs.c_global if c is not None ]
		l_c = np.hstack(l_c)
		linear_law.fit( law._Y , c_loc = l_c , c_scale = l_c )
		linear_loc   = linear_law.loc
		linear_scale = linear_law.scale
		
		coef = np.zeros(self.n_features)
		design = np.stack( (np.ones_like(l_c),l_c) , -1 ).squeeze()
		
		idxloc   = np.isfinite(np.log(linear_loc))
		idxscale = np.isfinite(np.log(linear_scale))
		resloc,_,_,_   = scl.lstsq( design[idxloc,:]   , np.log(linear_loc[idxloc]) )
		resscale,_,_,_ = scl.lstsq( design[idxscale,:] , np.log(linear_scale[idxscale]) )
		coef[0] = np.exp(resloc[0])
		coef[1] = np.exp(resscale[0])
		
		alphaloc   = resloc[1]   * coef[0]
		alphascale = resscale[1] * coef[0]
		coef[2]    = ( alphaloc + alphascale ) / 2
		
		return coef
	##}}}

