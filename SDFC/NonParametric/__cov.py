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

import numpy        as np
import scipy.linalg as scl

from ..link.__Univariate import ULIdentity


###############
## Functions ##
###############

def cov( Y0 , Y1 , c_Y = None , m_Y0 = None , m_Y1 = None , link = ULIdentity() , value = True ):
	"""
	SDFC.NonParametric.cov
	======================
	
	Estimate covariance given a covariate (or not)
	
	Parameters
	----------
	Y0    : np.array
		Dataset0 to fit the covariance between Y0 and Y1 
	Y1    : np.array
		Dataset1 to fit the covariance between Y0 and Y1 
	c_Y   : np.array or None
		Covariate(s)
	m_Y0  : np.array or float or None
		mean of Y0. If None, m_Y0 = np.mean(Y0)
	m_Y1  : np.array or float or None
		mean of Y1. If None, m_Y1 = np.mean(Y1)
	link  : class based on SDFC.tools.AbstractLink
		Link function, default is identity
	value : bool
		If true, return coefficients with covariates, else return covariance fitted
	
	Returns
	-------
	The covariance
	"""
	out,coef = None,None
	if c_Y is None:
		out  = np.cov( np.stack( (Y0.ravel(),Y1.ravel()) ) )[0,1]
		coef = link.inverse(out)
	else:
		m_Y0 = np.mean( Y0 ) if m_Y0 is None else np.array( [m_Y0] ).reshape(-1,1)
		m_Y1 = np.mean( Y1 ) if m_Y1 is None else np.array( [m_Y1] ).reshape(-1,1)
		
		Yres = ( Y0 - m_Y0 ) * ( Y1 - m_Y1 )
		if c_Y.ndim == 1: c_Y = c_Y.reshape(-1,1)
		design = np.hstack( ( np.ones( (Y0.size,1) ) , c_Y ) )
		coef,_,_,_ = scl.lstsq( design , link.inverse( Yres ) )
		out = link( design @ coef )
	
	return out if value else coef


