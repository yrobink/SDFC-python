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


###############
## Libraries ##
##{{{

import sys,os
import texttable as tt
import itertools as itt

## Scientific libraries
##=====================

import numpy as np
import scipy.stats as sc
import SDFC as sd
import SDFC.link as sdl

##}}}
###############


###############
## Fonctions ##
###############


#############
## Classes ##
#############

def global_lf_test( show = False , **kwargs ):
	
	## Parameters
	##===========
	n_samples   = kwargs["n_samples"]
	n_bootstrap = kwargs["n_bootstrap"]
	alpha       = kwargs["alpha"]
	_,X_loc,_,_ = sd.Dataset.covariates(n_samples)
	X_loc       = X_loc.reshape(-1,1)
	
	## Output tabular
	##===============
	tab = tt.Texttable( max_width = 0 )
	tab.header( ["Global Link Function ({} BS)".format(n_bootstrap),"Status","Max diff","True value","Estimated value","Quantile {}".format(alpha/2),"Quantile {}".format(1-alpha/2)] )
	
	## For Normal law
	##===============
	try:
		coef_ = np.array([0.5,1.2,-0.9])
		c_global = [X_loc]
		l_global = sdl.NormalRatioLocScaleConstant(n_samples)
		loc,scale = l_global.transform( coef_  , c_global )
		Y = np.random.normal( loc = loc , scale = scale )
		
		law = sd.Normal()
		law.fit_bootstrap( Y , l_global = l_global , c_global = c_global , **kwargs )
		out = ["NormalRatioLocScaleConstant","OK"]
		out.append( np.max( np.abs( coef_ - law.coef_ ) ) )
		out.append( np.round( coef_ , 2 ) )
		out.append( np.round( law.coef_ , 2 ) )
		out.append( np.round( law.info_.coefs_ci_bs_[0,:] , 2 ) )
		out.append( np.round( law.info_.coefs_ci_bs_[1,:] , 2 ) )
		tab.add_row(out)
	except:
		out = ["NormalRatioLocScaleConstant","Fail","/","/","/","/","/"]
		tab.add_row(out)
	
	## For GEV law
	##============
	try:
		coef_ = np.array([0.5,1.2,-0.9,-0.2])
		c_global = [X_loc]
		l_global = sdl.GEVRatioLocScaleConstant(n_samples)
		loc,scale,shape = l_global.transform( coef_  , c_global )
		Y = sc.genextreme.rvs( loc = loc , scale = scale , c = - shape )
		
		law = sd.GEV()
		law.fit_bootstrap( Y , l_global = l_global , c_global = c_global , **kwargs )
		out = ["GEVRatioLocScaleConstant","OK"]
		out.append( np.max( np.abs( coef_ - law.coef_ ) ) )
		out.append( np.round( coef_ , 2 ) )
		out.append( np.round( law.coef_ , 2 ) )
		out.append( np.round( law.info_.coefs_ci_bs_[0,:] , 2 ) )
		out.append( np.round( law.info_.coefs_ci_bs_[1,:] , 2 ) )
		tab.add_row(out)
	except:
		out = ["GEVRatioLocScaleConstant","Fail","/","/","/","/","/"]
		tab.add_row(out)
	
	if show:
		print(tab.draw())
	
	return tab


##########
## main ##
##########

if __name__ == "__main__":
	np.seterr( all = "ignore" )
	np.random.seed(42)
	
	kwargs = {}
	try:    kwargs["n_samples"]   = int(sys.argv[1])
	except: kwargs["n_samples"]   = 2000
	try:    kwargs["n_bootstrap"] = int(sys.argv[2])
	except: kwargs["n_bootstrap"] = 100
	try:    kwargs["alpha"]       = float(sys.argv[3])
	except: kwargs["alpha"]       = 0.1
	
	kwargs["show"] = "--show" in sys.argv
	
	with open( "test_customML.log" , "w" ) as f:
		tab = global_lf_test(**kwargs)
		f.write( tab.draw() + "\n" )
	
	
	print("Done")




