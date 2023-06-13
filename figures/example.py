# -*- coding: utf-8 -*-

#############################
## Yoann Robin             ##
## yoann.robin.k@gmail.com ##
#############################

###############
## Libraries ##
###############


import numpy as np
import pandas as pd

import SDFC as sd
import SDFC.NonParametric as sdnp

import matplotlib as mpl
try:
	import matplotlib.pyplot as plt
except:
	mpl.use("Qt5Agg")
	import matplotlib.pyplot as plt


####################
## Paramètres mpl ##
####################

mpl.rcParams['font.size'] = 15


###############
## Fonctions ##
###############


#############
## Classes ##
#############


##########
## main ##
##########

if __name__ == "__main__":
	
	np.random.seed(42)
	
	## Generate data
	size = 2500
	t = np.linspace( 0 , 1 , size )
	X0 = t**2
	X1 = np.cos( 2 * np.pi * t )
	
	loc   = 1. + 2 * X0
	scale = 0.6 + 0.5 * X1
	Y    = np.random.normal( loc = loc , scale = scale , size = size )
	
	
	## Fit a Gaussian law with MLE
	floc   = []
	fscale = []
	lmethods = ["moments","bayesian","mle"]
	for method in lmethods:
		law = sd.Normal(method = method)
		law.fit( Y , c_loc = X0 , c_scale = X1 )
		floc.append( law.loc )
		fscale.append( law.scale )
	
	
	## Fit a Quantile Regression
	probs = np.linspace( 0.01 , 0.99 , 100 )
	qr = sdnp.quantile( Y , probs , c_Y = np.vstack( (X0,X1) ).T )
	
	## Plot
	nrow,ncol = 1,2
	fs = 5
	fig = plt.figure( figsize = (fs*ncol,0.8*fs*nrow) )
	
	ax  = fig.add_subplot( nrow , ncol , 1 )
	ax.plot( t , Y , color = "blue" , linestyle = "" , marker = "." , alpha = 0.5 , label = r"$N(\mu(t),\sigma(t))$" )
	color = [ c for c in plt.cm.Reds( np.linspace( 0.3 , 0.7 , 3 ) ) ]
	for i,m in enumerate(lmethods):
		ax.plot( t , floc[i] , color = color[i] , label = m )
		ax.plot( t , floc[i] - fscale[i] , color = color[i] , linestyle = "--" )
		ax.plot( t , floc[i] + fscale[i] , color = color[i] , linestyle = "--" )
	
	ax.set_xlabel( r"$t$" )
	ax.set_title( "Gaussian Regression" )
	ax.legend( loc = "upper center" , fontsize = 10 )
	
	ax = fig.add_subplot( nrow , ncol , 2 )
	ax.plot( t , Y , color = "blue" , linestyle = "" , marker = "." , alpha = 0.5 )
	for i in range(100):
		ax.plot( t , qr[:,i] , color = "grey" , alpha = 0.5 )
	ax.set_xlabel( r"$t$" )
	ax.set_title( "Quantile Regression" )
	
	plt.tight_layout()
	plt.savefig( "Example_py.png" )
	
	print("Done")




