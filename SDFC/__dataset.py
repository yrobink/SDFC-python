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


#############
## Classes ##
#############

class Dataset:
	"""
		SDFC.Dataset
		============
		
		Some dataset to test SDFC
		
	"""
	def normal_cst_scale( size ):
		"""
		SDFC.Dataset.normal_cst_scale
		=============================
		
		
		Parameters
		----------
		size : int
			Length of dataset
		
		Returns
		-------
		t : np.array
			A "time" axis
		X : np.array
			A covariate
		Y :
			The dataset to fit
		
		"""
		t = np.linspace( 0 , 1 , size )
		X = t**2 + np.cos( 2* np.pi * t ) * 0.2
		Y = np.random.normal( loc = X , scale = np.repeat( 0.1 , size ) )
		return t,X,Y
	
	def covariates( size ):
		"""
		SDFC.Dataset.covariates
		=======================
		
		Return 3 covariates, one for loc, one for scale and one for shape.
		
		Parameters
		----------
		size : int
			Length of dataset
		
		Returns
		-------
		t       : np.array
			A "time" axis
		X_loc   : np.array
			A covariate for loc
		X_scale : np.array
			A covariate for loc
		X_shape : np.array
			A covariate for loc
		
		"""
		t = np.linspace( 0 , 1 , size )
		X_loc   = t**2 + np.cos( 2* np.pi * t ) * 0.2
		X_scale = 2 * t**2 - 2 * t + 1
		X_shape = 2 / ( 1 + np.exp( - 8 * (t-0.5) ) ) - 1
		return t,X_loc,X_scale,X_shape
