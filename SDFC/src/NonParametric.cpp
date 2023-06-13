
// Copyright(c) 2020 Yoann Robin
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

//=========//
// Include //
//=========//

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include "QuantileRegression.hpp"


//============//
// namespaces //
//============//

namespace py = pybind11 ;


//========//
// Module //
//========//

PYBIND11_MODULE( __NonParametric_cpp , m )
{
	//===========//
	// Functions //
	//===========//
	
	
	//=======//
	// Class //
	//=======//
	
	py::class_<QuantileRegression>( m , "QuantileRegression" )
	.def( py::init<>()                                              ) 
	.def( py::init<double>()                      , py::arg("ltau") )
	.def( py::init<py::list>()                    , py::arg("ltau") )
	.def( py::init<Eigen::Ref<Eigen::VectorXd>>() , py::arg("ltau") )
	.def( "__repr__"       , &QuantileRegression::repr )
	.def( "fit"            , &QuantileRegression::fit , py::arg("Y") , py::arg("X") )
	.def( "is_fitted"      , &QuantileRegression::is_fitted     )
	.def( "is_success"     , &QuantileRegression::is_success    )
	.def( "is_unfeasible"  , &QuantileRegression::is_unfeasible )
	.def( "set_fit_params" , &QuantileRegression::set_fit_params , py::arg("maxit") = 50 , py::arg("tol") = 1e-6 , py::arg("beta") = 0.99995 )
	.def( "set_ltau"       , (void (QuantileRegression::*) (double))                      &QuantileRegression::set_ltau , py::arg("ltau") )
	.def( "set_ltau"       , (void (QuantileRegression::*) (py::list))                    &QuantileRegression::set_ltau , py::arg("ltau") )
	.def( "set_ltau"       , (void (QuantileRegression::*) (Eigen::Ref<Eigen::VectorXd>)) &QuantileRegression::set_ltau , py::arg("ltau") )
	.def_readwrite( "coef_"     , &QuantileRegression::m_coef      )
	.def_readwrite( "quantiles" , &QuantileRegression::m_quantiles )
	;
	
	//============//
	// Attributes //
	//============//
	
//	m.doc() = "pybind11 example plugin" ; // optional module docstring
	m.attr("__name__") = "SDFC.NonParametric.__NonParametric_cpp";
//	m.attr("the_answer") = 42;
}
