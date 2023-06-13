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
from .__quantile import quantile


###############
## Functions ##
###############

def median( Y , c_Y = None , value = True ):
	"""
	SDFC.NonParametric.median
	=========================
	
	Estimate the median
	
	Parameters
	----------
	Y       : np.array
		Dataset to fit the median
	c_Y   : np.array or None
		Covariate(s)
	value : bool
		If true return value fitted, else return coefficients of fit
	
	Returns
	-------
	The median
	"""
	return quantile( Y , [0.5] , c_Y , value )



