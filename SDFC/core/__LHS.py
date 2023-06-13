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

###############
## Class(es) ##
###############

class LHS:
	def __init__( self , names : list , n_samples : int ):
		self.names     = names
		self.n_lhs     = len(self.names)
		self.n_samples = n_samples
		self._values   = { n : None for n in self.names }
		self.jacobian_ = None
		self._fixed    = { n : False for n in self.names }
	
	def is_fixed( self , name ):
		return self._fixed.get(name)
	
	@property
	def values_( self ):
		return self._values
	
	@values_.setter
	def values_( self , values ):
		for n,v in zip(self.names,values):
			self._values[n] = v
