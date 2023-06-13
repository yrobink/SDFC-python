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


class SDFCLawBootstrapTest:##{{{
	
	def __init__( self , **kwargs ): ##{{{
		self.name      = kwargs["name"]
		self.n_samples = kwargs["n_samples"]
		self.n_bootstrap = kwargs["n_bootstrap"]
		self.alpha     = kwargs["alpha"]
		self.sd_law    = kwargs["sd_law"]
		self.shape_p   = kwargs["shape_p"]
		t,X_loc,X_scale,X_shape = sd.Dataset.covariates(self.n_samples)
		self.t       = t
		self.X_loc   = X_loc.reshape(-1,1)
		self.X_scale = X_scale.reshape(-1,1)
		self.X_shape = X_shape.reshape(-1,1)
		self.has_loc   = "loc"   in kwargs["params"]
		self.has_scale = "scale" in kwargs["params"]
		self.has_shape = "shape" in kwargs["params"]
		self.n_params  = self.has_loc + self.has_scale + self.has_shape
		self.kwargs  = {}
		self.coef_   = []
	##}}}
	
	
	def build_loc( self , code ):##{{{
		coef_    = [0.5,1.]
		if code == 1:
			coef_      = [coef_[0]]
			self.loc   = np.repeat( coef_[0] , self.n_samples ).reshape(-1,1)
			self.coef_ = self.coef_ + coef_
		else:
			self.loc = coef_[0] + coef_[1] * self.X_loc
			if code == 0:
				self.kwargs["c_loc"] = self.X_loc
				self.coef_ = self.coef_ + coef_
			else:
				self.kwargs["f_loc"] = self.loc
	##}}}
	
	def build_scale( self , code ):##{{{
		coef_    = [0.3,-0.9]
		if code == 1:
			coef_      = [coef_[0]]
			self.scale = np.repeat( np.exp(coef_[0]) , self.n_samples ).reshape(-1,1)
			self.coef_ = self.coef_ + coef_
			self.kwargs["l_scale"] = sdl.ULExponential()
		else:
			self.scale = np.exp( coef_[0] + coef_[1] * self.X_scale )
			if code == 0:
				self.kwargs["c_scale"] = self.X_scale
				self.kwargs["l_scale"] = sdl.ULExponential()
				self.coef_ = self.coef_ + coef_
			else:
				self.kwargs["f_scale"] = self.scale
	##}}}
	
	def build_shape( self , code ):##{{{
		if self.shape_p:
			coef_ = [1,-0.2]
		else:
			coef_    = [0.,0.2]
		if code == 1:
			coef_      = [-coef_[1]]
			self.shape = np.repeat( coef_[0] , self.n_samples ).reshape(-1,1)
			self.coef_ = self.coef_ + coef_
		else:
			self.shape = coef_[0] + coef_[1] * self.X_shape
			if code == 0:
				self.kwargs["c_shape"] = self.X_shape
				self.coef_ = self.coef_ + coef_
			else:
				self.kwargs["f_shape"] = self.shape
	##}}}
	
	
	def testXXX( self , code , method = "MLE" ):##{{{
		
		i = 0
		if self.has_loc:
			self.build_loc( code[i] )
			i += 1
		if self.has_scale:
			self.build_scale( code[i] )
			i += 1
		if self.has_shape:
			self.build_shape( code[i] )
		
		self.Y = self.rvs()
		
		self.coef_ = np.array(self.coef_)
		if self.coef_.size > 0:
			self.kwargs["prior"] = sc.multivariate_normal( mean = self.coef_ , cov = 0.1 * np.identity(self.coef_.size) )
		self.law = self.sd_law( method = method )
		self.law.fit_bootstrap( self.Y , self.n_bootstrap , self.alpha , **self.kwargs )
	##}}}
	
	
	def summary( self , show = False ): ##{{{
		print( "## => {} / {} / {}".format( np.max(np.abs(self.coef_ - self.law.coef_)) , self.coef_ , np.round(self.law.coef_,3) ) )
	##}}}
	
	def run_all( self , method = "MLE" , show = True ):##{{{
		tab = tt.Texttable( max_width = 0 )
		tab.header( ["{} law test ({}, {} BS)".format(self.name,method,self.n_bootstrap),"Status","Max diff","True value","Estimated value","Quantile {}".format(self.alpha/2),"Quantile {}".format(1-self.alpha/2)] )
		for idx in itt.product( *(range(3) for _ in range(self.n_params))):
			str_idx = "".join(( str(i) for i in idx))
			try:
				self.kwargs = {}
				self.coef_  = []
				self.testXXX( idx , method )
				out = ["Test {}".format(str_idx),"OK",np.max(np.abs(self.coef_ - self.law.coef_))]
				out.append( self.coef_ )
				out.append( np.round(self.law.coef_,2) ) 
				out.append( np.round( self.law.info_.coefs_ci_bs_[0,:] ,2 ) )
				out.append( np.round( self.law.info_.coefs_ci_bs_[1,:] ,2 ) )
				tab.add_row( out )
			except NameError:
				if np.min(idx) == 2:
					tab.add_row( ["Test {}".format(str_idx),"OK","/","/","/","/","/"] )
				else:
					tab.add_row( ["Test {}".format(str_idx),"Fail","/","/","/","/","/"] )
			except:
				tab.add_row( ["Test {}".format(str_idx),"Fail","/","/","/","/","/"] )
		
		if show: print(tab.draw())
		return tab
	##}}}
	
##}}}


class NormalTest(SDFCLawBootstrapTest): ##{{{
	
	def __init__( self , n_bootstrap = 100 , alpha = 0.1 , n_samples = 2000 ): ##{{{
		kwargs = { "n_samples" : n_samples ,
				"n_bootstrap" : n_bootstrap ,
				"alpha"   : alpha , 
				"name"    : "Normal" ,
				"sd_law"  : sd.Normal ,
		        "params"  : ["loc","scale"] ,
		        "shape_p" : False
		        }
		SDFCLawBootstrapTest.__init__( self , **kwargs )
	
	##}}}
	
	def rvs( self ):##{{{
		return sc.norm.rvs( loc = self.loc , scale = self.scale )
	##}}}
	
##}}}

class ExponentialTest(SDFCLawBootstrapTest): ##{{{
	
	def __init__( self , n_bootstrap = 100 , alpha = 0.1 , n_samples = 2000 ): ##{{{
		kwargs = { "n_samples" : n_samples ,
				"n_bootstrap" : n_bootstrap ,
				"alpha"   : alpha , 
				"name"    : "Exponential" ,
				"sd_law"  : sd.Exponential ,
		        "params"  : ["scale"] ,
		        "shape_p" : False
		        }
		SDFCLawBootstrapTest.__init__( self , **kwargs )
	
	##}}}
	
	def rvs( self ):##{{{
		return sc.expon.rvs( scale = self.scale )
	##}}}
	
##}}}

class GammaTest(SDFCLawBootstrapTest): ##{{{
	
	def __init__( self , n_bootstrap = 100 , alpha = 0.1 , n_samples = 2000 ): ##{{{
		kwargs = { "n_samples" : n_samples ,
				"n_bootstrap" : n_bootstrap ,
				"alpha"   : alpha , 
				"name"    : "Gamma" ,
				"sd_law"  : sd.Gamma ,
		        "params"  : ["scale","shape"] ,
		        "shape_p" : True
		        }
		SDFCLawBootstrapTest.__init__( self , **kwargs )
	
	##}}}
	
	def rvs( self ):##{{{
		return np.random.gamma( scale = self.scale , shape = self.shape )
	##}}}
	
##}}}

class GEVTest(SDFCLawBootstrapTest): ##{{{
	
	def __init__( self , n_bootstrap = 100 , alpha = 0.1 , n_samples = 2000 ): ##{{{
		kwargs = { "n_samples" : n_samples ,
				"n_bootstrap" : n_bootstrap ,
				"alpha"   : alpha , 
				"name"    : "GEV" ,
				"sd_law"  : sd.GEV ,
		        "params"  : ["loc","scale","shape"] ,
		        "shape_p" : False
		        }
		SDFCLawBootstrapTest.__init__( self , **kwargs )
	
	##}}}
	
	def rvs( self ):##{{{
		return sc.genextreme.rvs( loc = self.loc , scale = self.scale , c = - self.shape )
	##}}}
	
##}}}

class GPDTest(SDFCLawBootstrapTest): ##{{{
	
	def __init__( self , n_bootstrap = 100 , alpha = 0.1 , n_samples = 2000 ): ##{{{
		kwargs = { "n_samples" : n_samples ,
				"n_bootstrap" : n_bootstrap ,
				"alpha"   : alpha , 
				"name"    : "GPD" ,
				"sd_law"  : sd.GPD ,
		        "params"  : ["scale","shape"] ,
		        "shape_p" : False
		        }
		SDFCLawBootstrapTest.__init__( self , **kwargs )
	
	##}}}
	
	def rvs( self ):##{{{
		return sc.genpareto.rvs( loc = self.loc , scale = self.scale , c = self.shape )
	##}}}
	
	def testXXX( self , code , method = "MLE" ): ##{{{
		self.loc = 1. + 0.5 * self.X_loc
		self.kwargs["f_loc"] = self.loc
		
		SDFCLawBootstrapTest.testXXX( self , code , method )
	##}}}
	
##}}}


##########
## main ##
##########

if __name__ == "__main__":
	np.seterr( all = "ignore" )
	np.random.seed(42)
	
	show = "--show" in sys.argv
	kwargs = {}
	try:    kwargs["n_samples"]   = int(sys.argv[1])
	except: kwargs["n_samples"]   = 2000
	try:    kwargs["n_bootstrap"] = int(sys.argv[2])
	except: kwargs["n_bootstrap"] = 100
	try:    kwargs["alpha"]       = float(sys.argv[3])
	except: kwargs["alpha"]       = 0.1
	
	
	with open( "test_bootstrap.log" , "w" ) as f:
		l_test = [NormalTest,ExponentialTest,GammaTest,GEVTest,GPDTest]
		for test in l_test:
			t = test(**kwargs)
			tab = t.run_all( "MLE" , show = show )
			f.write( tab.draw() + "\n" )
	
	
	
	print("Done")



