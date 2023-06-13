
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

#ifndef SDFC_NONPARAMETRIC_QUANTILEREGRESSION
#define SDFC_NONPARAMETRIC_QUANTILEREGRESSION

//-----------//
// Libraries //
//-----------//


#include <limits>
#include <cmath>
#include <random>
#include <pybind11/pybind11.h>
#include <Eigen/Dense>
#include <Eigen/Core>


#include "FrishNewton.hpp"

//============//
// namespaces //
//============//

namespace py = pybind11 ;

//=======//
// Class //
//=======//

struct QuantileRegression
{
	//=========//
	// Typedef // 
	//=========//
	
	//{{{
	typedef unsigned int    size_type  ;
	typedef double          value_type ;
	typedef Eigen::ArrayXd  Array      ;
	typedef Eigen::VectorXd Vector     ;
	typedef Eigen::MatrixXd Matrix     ;
	//}}}
	
	
	//=============//
	// Constructor //
	//=============//
	
	QuantileRegression(): //{{{
		m_ltau() ,
		m_coef() ,
		m_quantiles() ,
		m_frishNewton( 0.5 , 50 , 1e-6 , 0.99995 ),
		m_state(not_fitted)
	{} //}}}
	
	QuantileRegression( value_type ltau ): //{{{
		m_ltau(1) ,
		m_coef() ,
		m_quantiles() ,
		m_frishNewton( 0.5 , 50 , 1e-6 , 0.99995 ),
		m_state(not_fitted)
	{
		m_ltau[0] = ltau ;
	} //}}}
	
	QuantileRegression( py::list ltau ): //{{{
		m_ltau(py::len(ltau)) ,
		m_coef() ,
		m_quantiles() ,
		m_frishNewton( 0.5 , 50 , 1e-6 , 0.99995 ),
		m_state(not_fitted)
	{
		int s = 0 ;
		for( auto item : ltau )
			m_ltau[s++] = item.cast<value_type>() ;
	} //}}}
	
	QuantileRegression( Eigen::Ref<Vector> ltau ): //{{{
		m_ltau(ltau) ,
		m_coef() ,
		m_quantiles() ,
		m_frishNewton( 0.5 , 50 , 1e-6 , 0.99995 ),
		m_state(not_fitted)
	{} //}}}
	
	~QuantileRegression() //{{{
	{} //}}}
	
	std::string repr()//{{{
	{
		std::string True("True"), False("False") ;
		std::string _repr("") ;
		_repr += "SDFC.NonParametric.QuantileRegression\n" ;
		_repr += "=====================================\n" ;
		_repr += "* Method    : Frish-Newton\n" ;
		_repr += "* Fitted    : " + ( is_fitted()     ? True : False ) + "\n" ;
		_repr += "* Success   : " + ( is_success()    ? True : False ) + "\n" ;
		_repr += "* Unfeasible: " + ( is_unfeasible() ? True : False ) + "\n" ;
		return _repr ;
	}
	//}}}
	
	
	//===========//
	// Accessors //
	//===========//
	
	void set_fit_params( size_type maxit , value_type tol , value_type beta ) //{{{
	{
		m_frishNewton.m_maxit = maxit ;
		m_frishNewton.m_tol   = tol ;
		m_frishNewton.m_beta  = beta ;
	}
	//}}}
	
	void set_ltau( double ltau ) //{{{
	{
		m_ltau.resize(1) ;
		m_ltau[0] = ltau ;
	} //}}}
	
	void set_ltau( py::list ltau ) //{{{
	{
		m_ltau.resize( py::len(ltau) ) ;
		int s = 0 ;
		for( auto item : ltau )
			m_ltau[s++] = item.cast<value_type>() ;
	} //}}}
	
	void set_ltau( Eigen::Ref<Vector> ltau ) //{{{
	{
		m_ltau = ltau ;
	} //}}}
	
	Matrix coef() //{{{
	{ return m_coef ; } //}}}
	
	qrstate_t state() //{{{
	{ return m_state ; } //}}}
	
	
	//=======//
	// State //
	//=======//
	
	bool is_fitted()//{{{
	{ return ( m_state == success || m_state == unfeasible ) ; }//}}}
	
	bool is_success()//{{{
	{ return m_state == success ; }//}}}
	
	bool is_unfeasible()//{{{
	{ return m_state == unfeasible ; }//}}}
	
	
	//=========//
	// Methods //
	//=========//
	
	void fit( Vector& Y , Matrix& X ) //{{{
	{
		size_type n_tau = m_ltau.size() ;
		size_type n_cov = X.cols() + 1 ;
		m_coef.resize( n_tau , n_cov ) ;
		for( size_type i = 0 ; i < n_tau ; ++i )
		{
			m_frishNewton.set_tau( m_ltau[i] ) ;
			m_frishNewton.fit( Y , X ) ;
			m_state = m_frishNewton.state() ;
			if( m_state == unfeasible )
				break ;
			m_coef.row(i) = m_frishNewton.m_coef ;
		}
		m_quantiles = predict() ;
	} //}}}
	
	Matrix predict() //{{{
	{
		size_type n_tau = m_ltau.size() ;
		Matrix& A(m_frishNewton.m_A) ;
		Matrix Yq( A.rows() , n_tau ) ;
		Yq = A * m_coef.transpose() ;
		return Yq ;
	} //}}}
	
	
	//===========//
	// Arguments //
	//===========//
	
	//{{{
	Array       m_ltau        ;
	Matrix      m_coef        ;
	Matrix      m_quantiles   ;
	FrishNewton m_frishNewton ;
	qrstate_t   m_state       ;
	//}}}
	
	
	
};

#endif

