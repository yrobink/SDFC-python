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
from .__NonParametric_cpp  import QuantileRegression


###############
## Functions ##
###############

def quantile( Y , ltau , c_Y = None , value = True ):
	"""
	SDFC.NonParametric.quantile
	===========================
	
	Estimate quantile given a covariate (or not)
	
	Parameters
	----------
	Y       : np.array
		Dataset to fit the quantile
	ltau    : np.array
		The quantile to fit, between 0 and 1
	c_Y   : np.array or None
		Covariate(s)
	link  : class based on SDFC.tools.Link
		Link function, default is identity
	value : bool
		If true return value fitted, else return coefficients of fit
	
	Returns
	-------
	The quantiles
	"""
	
	ltau = np.array( [ltau] ).ravel()
	q    = None
	coef = None
	
	if c_Y is None:
		q    = np.percentile( Y , 100 * ltau )
		coef = q.copy()
	else:
		reg  = QuantileRegression( ltau = ltau )
		reg.fit( Y , c_Y )
		q    = reg.quantiles
		coef = reg.coef_
	return q if value else coef


