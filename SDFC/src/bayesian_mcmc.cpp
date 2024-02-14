
// Copyright(c) 2024 Yoann Robin
// 
// This file is part of SDFC.
// 
// SDFC is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// SDFC is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with SDFC.  If not, see <https://www.gnu.org/licenses/>.


//-----------//
// Libraries //
//-----------//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <Eigen/Core>

#include <cmath>
#include <random>
#include <tuple>


//============//
// namespaces //
//============//

namespace py = pybind11 ;


std::tuple<Eigen::MatrixXd,std::vector<bool>> mcmc_cpp( Eigen::Ref<const Eigen::VectorXd> init ,
                      int n_drawn ,
                      std::function<double(Eigen::Ref<const Eigen::VectorXd>)> nlll ,
                      std::function<double(Eigen::Ref<const Eigen::VectorXd>)> prior,
                      std::function<Eigen::VectorXd(Eigen::Ref<const Eigen::VectorXd>)> transition
        )
{
	// Init output
	Eigen::MatrixXd draw( n_drawn , init.size() ) ;
	std::vector<bool> accept(n_drawn) ;
	
	// Init values
	double lll_current   = - nlll(init) ;
	double prior_current =  prior(init) ;
	double p_current     = prior_current + lll_current ;
	double lll_next   = 0 ;
	double prior_next = 0 ;
	double p_next     = 0 ;
	draw.row(0) = init;
	accept[0] = true ;
	
	// Init uniform distribution
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution(0.0,1.0);
	
	// Loop
	for( int i = 1 ; i < n_drawn ; ++i )
	{
		// Transition
		draw.row(i) = transition(draw.row(i-1)) ;
		
		// Likelihood and probability of new points
		lll_next   = - nlll(draw.row(i)) ;
		prior_next = prior(draw.row(i)) ;
		p_next     = prior_next + lll_next ;
		
		// Accept or not
		double u = distribution(generator) ;
		double p_accept = std::exp( p_next - p_current ) ;
		if( u < p_accept )
		{
			lll_current   = lll_next ;
			prior_current = prior_next ;
			p_current     = p_next ;
			accept[i] = true ;
		}
		else
		{
			draw.row(i) = draw.row(i-1) ;
			accept[i] = false ;
		}
	}
	
	return std::make_tuple(draw,accept) ;
}

//========//
// Module //
//========//

PYBIND11_MODULE( __bayesian_mcmc_cpp , m )
{
	//===========//
	// Functions //
	//===========//
	
	m.def( "mcmc_cpp" , &mcmc_cpp );
	
	
	//============//
	// Attributes //
	//============//
	
	m.attr("__name__") = "SDFC.bayesian_mcmc_cpp";
}

