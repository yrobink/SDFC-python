# -*- coding: utf-8 -*-

## Copyright(c) 2020 / 2025 Yoann Robin
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

import os
import sysconfig
import setuptools
from setuptools import setup
from setuptools import Extension
from pathlib import Path

import pybind11

############################
## Python path resolution ##
############################

cpath = Path(__file__).parent


####################################################
## Some class and function to compile with Eigen  ##
####################################################

class EigenNotFoundError(Exception):
	def __init__( self , *args , **kwargs ):
		super().__init__( *args , **kwargs )

def get_eigen_include():##{{{
	possible_path = []
	
	## Priority 1: custom user installation
	if os.environ.get("EIGEN_INCLUDE_PATH") is not None:
		possible_path.append( os.environ["EIGEN_INCLUDE_PATH"] )
	
	## Priority 2: user installation
	if os.environ.get("HOME") is not None:
		possible_path.append( os.path.join( os.environ["HOME"] , ".local/include" ) )
	
	## Priority 3: conda installation
	if os.environ.get("CONDA_PREFIX") is not None:
		possible_path.append( os.path.join( os.environ["CONDA_PREFIX"] , "include" ) )
	
	## Priority >3: others installations
	for p in [ os.path.dirname(sysconfig.get_paths()['include']), "/usr/include/" , "/usr/local/include/" ]:
		possible_path.append(p)
	
	## Check if eigen in this path
	for path in possible_path:
		
		eigen_include = os.path.join( path , "Eigen" )
		if os.path.isdir( eigen_include ):
			return path
		
		eigen_include = os.path.join( path , "eigen3" , "Eigen" )
		if os.path.isdir( eigen_include ):
			return os.path.join( path , "eigen3" )
	
	raise EigenNotFoundError("Eigen not found in all possible path, try to install if from conda or set the environement variable 'EIGEN_INCLUDE_PATH' to the path to Eigen")
##}}}


##########################
## Extension to compile ##
##########################

ext_modules = [
	Extension(
		'SDFC.NonParametric.__NonParametric_cpp',
		[ 'SDFC/src/NonParametric.cpp' ],
		include_dirs=[
			get_eigen_include(),
			pybind11.get_include(True),
			pybind11.get_include(False),
		],
		language='c++',
		depends = [
			"SDFC/src/QuantileRegression.hpp",
			"SDFC/src/FrishNewton.hpp"
			],
		extra_compile_args = ["-O3"]
	),
	Extension(
		'SDFC.__bayesian_mcmc_cpp',
		[ 'SDFC/src/bayesian_mcmc.cpp' ],
		include_dirs=[
			get_eigen_include(),
			pybind11.get_include(True),
			pybind11.get_include(False),
		],
		language='c++',
		extra_compile_args = ["-O3"],
	),
]


#################
## Compilation ##
#################

list_packages = setuptools.find_packages()
package_dir = { "SDFC": "SDFC" }


########################
## Infos from release ##
########################

release = {}
exec( (cpath / "SDFC" / "__release.py").read_text() , {} , release )


#################
## Description ##
#################
long_description = (cpath / "README.md").read_text()


#######################
## And now the setup ##
#######################

setup(
	name         = release['name'],
	description  = release['description'],
	long_description = long_description,
	long_description_content_type = 'text/markdown',
	version      = release['version'],
	author       = release['author'],
	author_email = release['author_email'],
	license      = release['license'],
	platforms        = [ "linux" , "macosx" ],
	classifiers      = [
		"Development Status :: 5 - Production/Stable",
		"Natural Language :: English",
		"Operating System :: MacOS :: MacOS X",
		"Operating System :: POSIX :: Linux",
		"Programming Language :: Python :: 3",
		"Programming Language :: Python :: 3.8",
		"Programming Language :: Python :: 3.9",
		"Programming Language :: Python :: 3.10",
		"Programming Language :: Python :: 3.11",
		"Programming Language :: Python :: 3.12",
		"Programming Language :: Python :: 3.13",
		"Topic :: Scientific/Engineering :: Mathematics"
	],
	ext_modules      = ext_modules,
	setup_requires   = ["pybind11>=2.2"],
	build_requires   = ["pybind11>=2.2"],
	install_requires = [ "numpy" , "scipy" , "pybind11>=2.2" ],
	zip_safe         = False,
	packages         = list_packages,
	package_dir      = package_dir
)



