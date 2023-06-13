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
from .__quantile import quantile


###############
## Functions ##
###############

def lmoments_matrix( size ):##{{{
	"""
		SDFC.NonParametric.lmoments_matrix
		==================================
		
		Build a matrix to infer L-Moments in stationary case. If M = lmoments_matrix(Y.size), then
		the fourth first L-Moments are just M.T @ np.sort(Y)
		
	"""
	C0 = scs.binom( range( size ) , 1 )
	C1 = scs.binom( range( size - 1 , -1 , -1 ) , 1 )
	
	## Order 3
	C2 = scs.binom( range( size ) , 2 )
	C3 = scs.binom( range( size - 1 , -1 , -1 ) , 2 )
	
	## Order 4
	C4 = scs.binom( range( size ) , 3 )
	C5 = scs.binom( range( size - 1 , -1 , -1 ) , 3 )
	
	M = np.zeros( (size,4) )
	M[:,0] = 1. / size
	M[:,1] = ( C0 - C1 ) / ( 2 * scs.binom( size , 2 ) )
	M[:,2] = ( C2 - 2 * C0 * C1 + C3 ) / ( 3 * scs.binom( size , 3 ) )
	M[:,3] = ( C4 - 3 * C2 * C1 + 3 * C0 * C3 - C5 ) / ( 4 * scs.binom( size , 4 ) )
	
	return M
##}}}

def _lmoments_stationary( Y ):##{{{
	Ys = np.sort(Y.squeeze())
	M = lmoments_matrix( Y.size )
	return M.T @ Ys
##}}}

def lmoments( Y , c_Y = None , order = None , lq = np.arange( 0.05 , 0.96 , 0.01 ) ):##{{{
	"""
	SDFC.NonParametric.lmoments
	===========================
	
	Estimate the lmoments of orders 1 to 4. If a covariate is given, a quantile regression is performed
	and the instantaneous L-Moments are estimated from the quantile fitted.
	
	Parameters
	----------
	Y     : np.array
		Dataset to fit the lmoments
	c_Y   : np.array or None
		Covariate
	order : integer, list of integer or None
		Integers between 1 and 4
	lq    : np.array
		Quantiles for quantile regression, only used if a covariate is given. Default is np.arange(0.05,0.96,0.01)
	
	Returns
	-------
	The lmoments.
	"""
	
	order = order if order is None else np.array( [order] , dtype = np.int ).squeeze() - 1
	
	if c_Y is None:
		lmom = _lmoments_stationary(Y)
		return lmom if order is None else lmom[order]
	else:
		Y = Y.reshape(-1,1)
		if c_Y.ndim == 1: c_Y = c_Y.reshape(-1,1)
		Yq = quantile( Y , lq , c_Y )
		M  = lmoments_matrix(Yq.shape[1])
		lmom = np.transpose( M.T @ Yq.T )
		if order is None:
			return lmom
		else:
			return lmom[:,order]
##}}}


