/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

/*! \file */
#ifndef LBM_H
#define LBM_H

#include "cppspec.h"

#if (N_DIM==2)
	#define l_dq 9							///< Number of particle velocity vectors in the discretized particle velocity set.
									///< When @ref N_DIM = 2, this is equal to 9 (i.e. D2Q9). In 3D, this value is set to 19 by default (i.e. D3Q19) but can be set to 27 for D3Q27 as well.
	#define l_dq_max 9						///< Number of particle velocity vectors requried for complete connectivity.
									///< This value is equal to @ref l_dq in 2D but equal to the D3Q27 amount in 3D (whereas @ref l_dq can take on separate values for D3Q19 and D3Q27).
#else
	#define l_dq 27							///< Number of particle velocity vectors in the discretized particle velocity set.
									///< When @ref N_DIM = 2, this is equal to 9 (i.e. D2Q9). In 3D, this value is set to 19 by default (i.e. D3Q19) but can be set to 27 for D3Q27 as well.
	#define l_dq_max 27						///< Number of particle velocity vectors requried for complete connectivity.
									///< This value is equal to @ref l_dq in 2D but equal to the D3Q27 amount in 3D (whereas @ref l_dq can take on separate values for D3Q19 and D3Q27).
#endif

#if (N_PRECISION==1)
	#define cs   	0.5773502691896258				///< Macro definition for lattice speed of sound (1/sqrt(3)).
	#define cs2  	0.3333333333333333 				///< Macro definition for @ref cs squared.
	#define cs4  	0.1111111111111111				///< Macro definition for @ref cs2 squared.
	#define cso2 	3.0000000000000000				///< Macro definition for inverse of @ref cs2.
	#define hcso2	1.5000000000000000				///< Macro definition for half of @ref cso2.
	#define hcso4	4.5000000000000000				///< Macro definition for half of inverse of @ref cs4.
#else
	#define cs   	0.5773503F					///< Macro definition for lattice speed of sound (1/sqrt(3)).
	#define cs2  	0.3333333F					///< Macro definition for @ref cs squared.
	#define cs4  	0.1111111F					///< Macro definition for @ref cs2 squared.
	#define cso2 	3.0000000F					///< Macro definition for inverse of @ref cs2.
	#define hcso2	1.5000000F					///< Macro definition for half of @ref cso2.
	#define hcso4	4.5000000F					///< Macro definition for half of inverse of @ref cs4.
#endif

//! Array of Gauss-Hermite quadrature weights for corresponding discrete particle velocity set.
const double	w[2][27] = 
{{4.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0},
#if (l_dq==19)
{1.0/3.0,1.0/18.0,1.0/18.0,1.0/18.0,1.0/18.0,1.0/18.0,1.0/18.0,1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0}};
#else
{8.0/27.0,2.0/27.0,2.0/27.0,2.0/27.0,2.0/27.0,2.0/27.0,2.0/27.0,1.0/54.0,1.0/54.0,1.0/54.0,1.0/54.0,1.0/54.0,1.0/54.0,1.0/54.0,1.0/54.0,1.0/54.0,1.0/54.0,1.0/54.0,1.0/54.0,1.0/216.0,1.0/216.0,1.0/216.0,1.0/216.0,1.0/216.0,1.0/216.0,1.0/216.0,1.0/216.0}};
#endif

//! Array of velocity vectors for corresponding discrete particle velocity set.
const double	c[2][27*3] =
{{ 0,1,0,-1,0,1,-1,-1,1,   0,0,1,0,-1,1,1,-1,-1},
{ 0,1,-1,0,0,0,0,1,-1,1,-1,0,0,1,-1,1,-1,0,0,1,-1,1,-1,1,-1,-1,1,   0,0,0,1,-1,0,0,1,-1,0,0,1,-1,-1,1,0,0,1,-1,1,-1,1,-1,-1,1,1,-1,   0,0,0,0,0,1,-1,0,0,1,-1,1,-1,0,0,-1,1,-1,1,1,-1,-1,1,1,-1,1,-1 }};

//! Array of indices for reflected velocity vectors for corresponding discrete particle velocity set.
const int	pb[2][27] =
{{0,3,4,1,2,7,8,5,6},
{0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15,18,17,20,19,22,21,24,23,26,25}};

// NOTE: Access c_id = c[N_DIM-2][p + d*l_dq_max];

#endif
