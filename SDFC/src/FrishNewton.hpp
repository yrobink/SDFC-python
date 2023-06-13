
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

//==============================================================================//
// This software is based on the fortran code developped by Roger Koenker,      //
// coming from the R package "quantreg", see:                                   //
// https://cran.r-project.org/web/packages/quantreg/index.html                  //
//==============================================================================//

//==============================================================================//
// Ce programme est basé sur le code fortran développé par Roger Koenker,       //
// venant du package R "quantreg", voir:                                        //
// https://cran.r-project.org/web/packages/quantreg/index.html                  //
//==============================================================================//


#ifndef SDFC_NONPARAMETRIC_FRISHNEWTON
#define SDFC_NONPARAMETRIC_FRISHNEWTON

//-----------//
// Libraries //
//-----------//


#include <limits>
#include <cmath>
#include <random>

#include <Eigen/Dense>
#include <Eigen/Core>


//=======//
// Class //
//=======//

enum qrstate_t //{{{
{
	success = 0,
	unfeasible = 1,
	not_fitted = 2
} ;
//}}}

struct Xi //{{{
{
	typedef unsigned int size_type  ;
	typedef double       value_type ;
	
	Xi():
		y(),
		z(),
		x(),
		s(),
		w()
	{}
	
	Xi( size_type p , size_type n ):
		y(p),
		z(n),
		x(n),
		s(n),
		w(n)
	{}
	
	~Xi()
	{}
	
	
	value_type mu()
	{
		return value_type(x.matrix().transpose() * z.matrix()) + value_type(s.matrix().transpose() * w.matrix()) ;
	}
	
	Eigen::ArrayXd y ;
	Eigen::ArrayXd z ;
	Eigen::ArrayXd x ;
	Eigen::ArrayXd s ;
	Eigen::ArrayXd w ;
} ;
// }}}

struct QuantileRegression ;

class FrishNewton
{
	friend QuantileRegression ;
	public:
	
	//=========//
	// Typedef // 
	//=========//
	
	//{{{
	typedef unsigned int    size_type          ;
	typedef double          value_type         ;
	typedef Eigen::ArrayXd  Array              ;
	typedef Eigen::VectorXd Vector             ;
	typedef Eigen::MatrixXd Matrix             ;
	typedef Eigen::FullPivLU<Matrix> PLUMatrix ;
	//}}}
	
	//=============//
	// Constructor //
	//=============//
	
	FrishNewton( value_type tau , size_type maxit = 50 , value_type tol = 1e-6 , value_type beta = 0.99995 ): //{{{
		m_tau(tau) ,
		m_maxit(maxit),
		m_tol(tol),
		m_beta(beta),
		m_p(),
		m_n(),
		m_A(),
		m_lu(),
		m_iAQA(),
		m_xi(),
		m_dxi(),
		m_alphaP(),
		m_alphaD(),
		m_mu(0),
		m_b(),
		m_c(),
		m_iq(),
		m_u(),
		m_dr(),
		m_rhs(),
		m_coef(),
		m_state(not_fitted)
	{} //}}}
	
	~FrishNewton() //{{{
	{} //}}}
	
	//===========//
	// Accessors //
	//===========//
	
	void set_tau( value_type tau ) //{{{
	{
		m_tau = tau ;
	}//}}}
	
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
		
		initialize( Y , X ) ;
		if( m_state == unfeasible )
			return ;
		
		size_type nit = 0 ;
		while( m_mu > m_tol && ++nit < m_maxit )
		{
			predictor_step() ;
			if( m_state == unfeasible )
				return ;
			infer_gap() ;
			corrector_step() ;
			update() ;
		}
		m_state = success ;
		m_coef = - m_xi.y ;
		
	} //}}}
	
	Array predict() //{{{
	{
		return m_A * m_coef.matrix() ;
	} //}}}
	
	
	private:
	
	//=========//
	// Methods //
	//=========//
	
	void initialize( Vector& Y , Matrix& X ) //{{{
	{
		m_n = Y.size() ;
		m_p = X.cols() + 1 ;
		m_xi  = Xi(m_p,m_n) ;
		m_dxi = Xi(m_p,m_n) ;
		m_c   = -Y ;
		m_iq  = Array(m_n) ;
		m_u   = Array(m_n) ;
		m_dr  = Array(m_n) ;
		m_rhs = Array(m_p) ;
		
		// Design matrix
		m_A = Matrix(m_n,m_p) ;
		m_A.col(0).array() = 1. ;
		m_A.block( 0 , 1 , m_n , m_p - 1 ) = X ;
		
		//
		m_b = ( 1. - m_tau ) * m_A.colwise().sum() ;
		m_lu = PLUMatrix(m_A.transpose() * m_A) ;
		if( !m_lu.isInvertible() )
		{
			m_state = unfeasible ;
			return ;
		}
		m_iAQA = m_lu.inverse() ;
		m_iq = 1. ;
		m_u  = 1. ;
		m_xi.x = 1. - m_tau ;
		m_xi.y = m_A.transpose() * m_c.matrix() ;
		m_xi.y = m_iAQA * m_xi.y.matrix() ;
		m_xi.s = m_c - (m_A * m_xi.y.matrix()).array() ;
		
		for( size_type i = 0 ; i < m_n ; ++i )
		{
			m_xi.z[i] = std::max(  m_xi.s[i] , 0. ) + ( (std::abs(m_xi.s[i]) < m_tol) ? m_tol : 0 ) ;
			m_xi.w[i] = std::max( -m_xi.s[i] , 0. ) + ( (std::abs(m_xi.s[i]) < m_tol) ? m_tol : 0 ) ;
		}
		m_xi.s = m_u - m_xi.x ;
		m_mu = m_xi.mu() ;
	} //}}}
	
	void predictor_step() //{{{
	{
		for( size_type i = 0 ; i < m_n ; ++i )
		{
			m_iq[i] = 1. / ( m_xi.z[i] / m_xi.x[i] + m_xi.w[i] / m_xi.s[i] ) ;
			m_dxi.s[i] = m_xi.z[i] - m_xi.w[i] ;
			m_dxi.z[i] = m_iq[i] * m_dxi.s[i] ;
		}
		
		m_lu = PLUMatrix(m_A.transpose() * m_iq.matrix().asDiagonal() * m_A) ;
		if( !m_lu.isInvertible() )
		{
			m_state = unfeasible ;
			return ;
		}
		m_iAQA = m_lu.inverse() ;
		m_rhs = m_b - (m_A.transpose() * m_xi.x.matrix()).array() ;
		m_rhs += (m_A.transpose() * m_dxi.z.matrix()).array() ;
		m_dxi.y = m_iAQA * m_rhs.matrix() ;
		m_dxi.s = ( m_A * m_dxi.y.matrix() ).array() - m_dxi.s ;
		
		m_alphaP = std::numeric_limits<value_type>::max() ;
		m_alphaD = std::numeric_limits<value_type>::max() ;
		
		for( size_type i = 0 ; i < m_n ; ++i )
		{
			m_dxi.x[i] = m_iq[i] * m_dxi.s[i] ;
			m_dxi.s[i] = - m_dxi.x[i] ;
			m_dxi.z[i] = - m_xi.z[i] * ( m_dxi.x[i] / m_xi.x[i] + 1. ) ;
			m_dxi.w[i] = - m_xi.w[i] * ( m_dxi.s[i] / m_xi.s[i] + 1. ) ;
			
			if( m_dxi.x[i] < 0 )
				m_alphaP = std::min( m_alphaP , - m_xi.x[i] / m_dxi.x[i] ) ;
			if( m_dxi.s[i] < 0 )
				m_alphaP = std::min( m_alphaP , - m_xi.s[i] / m_dxi.s[i] ) ;
			if( m_dxi.z[i] < 0 )
				m_alphaD = std::min( m_alphaD , - m_xi.z[i] / m_dxi.z[i] ) ;
			if( m_dxi.w[i] < 0 )
				m_alphaD = std::min( m_alphaD , - m_xi.w[i] / m_dxi.w[i] ) ;
		}
		m_alphaP = std::min( m_beta * m_alphaP , 1. ) ;
		m_alphaD = std::min( m_beta * m_alphaD , 1. ) ;
	}//}}}
	
	void infer_gap() //{{{
	{
		m_mu = m_xi.mu() ;
		value_type mu_new = 0 ;
		for( size_type i = 0 ; i < m_n ; ++i )
		{
			mu_new += (m_xi.x[i] + m_alphaP * m_dxi.x[i]) * (m_xi.z[i] + m_alphaD * m_dxi.z[i]) + (m_xi.s[i] + m_alphaP * m_dxi.s[i]) * (m_xi.w[i] + m_alphaD * m_dxi.w[i]) ;
		}
		m_mu = std::pow( mu_new , 3 ) / std::pow( m_mu , 2 ) / ( 2. * static_cast<value_type>(m_n) ) ;
	}//}}}
	
	void corrector_step() //{{{
	{
		for( size_type i = 0 ; i < m_n ; ++i )
		{
			m_dr[i] = m_iq[i] * ( m_mu * ( 1. / m_xi.s[i] - 1. / m_xi.x[i] ) + m_dxi.x[i] * m_dxi.z[i] / m_xi.x[i] - m_dxi.s[i] * m_dxi.w[i] / m_xi.s[i] ) ;
		}
		std::swap( m_dxi.y , m_rhs ) ;
		m_dxi.y += ( m_A.transpose() * m_dr.matrix() ).array() ;
		m_dxi.y = m_iAQA * m_dxi.y.matrix() ;
		m_u = m_A * m_dxi.y.matrix() ;
		
		m_alphaP = std::numeric_limits<value_type>::max() ;
		m_alphaD = std::numeric_limits<value_type>::max() ;
		
		value_type dxdz, dsdw ;
		for( size_type i = 0 ; i < m_n ; ++i )
		{
			dxdz = m_dxi.x[i] * m_dxi.z[i] ;
			dsdw = m_dxi.s[i] * m_dxi.w[i] ;
			m_dxi.x[i] = m_iq[i] * ( m_u[i] - m_xi.z[i] + m_xi.w[i] ) - m_dr[i] ;
			m_dxi.s[i] = - m_dxi.x[i] ;
			m_dxi.z[i] = - m_xi.z[i] + ( m_mu - m_xi.z[i] * m_dxi.x[i] - dxdz ) / m_xi.x[i] ;
			m_dxi.w[i] = - m_xi.w[i] + ( m_mu - m_xi.w[i] * m_dxi.s[i] - dsdw ) / m_xi.s[i] ;
			
			if( m_dxi.x[i] < 0 )
				m_alphaP = std::min( m_alphaP , - m_xi.x[i] / m_dxi.x[i] ) ;
			if( m_dxi.s[i] < 0 )
				m_alphaP = std::min( m_alphaP , - m_xi.s[i] / m_dxi.s[i] ) ;
			if( m_dxi.z[i] < 0 )
				m_alphaD = std::min( m_alphaD , - m_xi.z[i] / m_dxi.z[i] ) ;
			if( m_dxi.w[i] < 0 )
				m_alphaD = std::min( m_alphaD , - m_xi.w[i] / m_dxi.w[i] ) ;
		}
		m_alphaP = std::min( m_beta * m_alphaP , 1. ) ;
		m_alphaD = std::min( m_beta * m_alphaD , 1. ) ;
	}//}}}
	
	void update() //{{{
	{
		m_xi.x += m_alphaP * m_dxi.x ;
		m_xi.s += m_alphaP * m_dxi.s ;
		m_xi.y += m_alphaD * m_dxi.y ;
		m_xi.z += m_alphaD * m_dxi.z ;
		m_xi.w += m_alphaD * m_dxi.w ;
		m_mu = m_xi.mu() ;
	}//}}}
	
	
	//===========//
	// Arguments //
	//===========//
	
	//{{{
	value_type m_tau              ;
	size_type  m_maxit            ;
	value_type m_tol              ;
	value_type m_beta             ;
	size_type  m_p                ;
	size_type  m_n                ;
	Matrix     m_A                ;
	PLUMatrix  m_lu               ;
	Matrix     m_iAQA             ;
	Xi         m_xi               ;
	Xi         m_dxi              ;
	value_type m_alphaP           ;
	value_type m_alphaD           ;
	value_type m_mu               ;
	Array      m_b                ;
	Array      m_c                ;
	Array      m_iq               ;
	Array      m_u                ;
	Array      m_dr               ;
	Array      m_rhs              ;
	Array      m_coef             ;
	qrstate_t  m_state            ;
	//}}}
	

} ;


#endif

