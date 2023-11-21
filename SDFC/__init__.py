# -*- coding: utf-8 -*-

## Copyright(c) 2020 / 2023 Yoann Robin
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


#############
## Imports ##
#############

from .__release import version
__version__ = version

from .__Normal      import Normal
from .__Exponential import Exponential
from .__Gamma       import Gamma
from .__GEV         import GEV
from .__GPD         import GPD

from .__dataset import Dataset
from .__AddonsMCMC import *
