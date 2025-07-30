#ifndef CUSTOM_H
#define CUSTOM_H

// o====================================================================================
// | Specified refinement region.
// o====================================================================================

template <typename ufloat_t>
__device__
bool Cu_RefineRegion
(
	const int L, const ufloat_t x, const ufloat_t y, const ufloat_t z, const ufloat_t xp, const ufloat_t yp, const ufloat_t zp,
	const int onb, const int nbr_1, const int nbr_2, const int nbr_3, const int nbr_4, const int nbr_5, const int nbr_6
)
{
	// --- HERE ---
	// Define a custom region of refinement here.
	// --- HERE ---
	//return (x > (ufloat_t)(2.0) && x < (ufloat_t)(0.5) && y > (ufloat_t)(0.3) && y < (ufloat_t)(0.7));
	double D = (double)(1<<L);
	return   x > 0.3-0.1/D && x <= 0.3+0.1/D && y > 0.5-0.25/D && y < 0.5+0.25/D; 
	
	//return false;
}

// o====================================================================================
// | Initial conditions.
// o====================================================================================

template <typename ufloat_t>
__device__ __forceinline__
void Cu_ComputeIC(ufloat_t &rho, ufloat_t &u, ufloat_t &v, ufloat_t &w, const ufloat_t x, const ufloat_t y, const ufloat_t z)
{
	// --- HERE ---
	// Change the initial conditions here.
	// --- HERE ---
	
	rho = (ufloat_t)1.0;
	u = (ufloat_t)0.05;
	v = (ufloat_t)0.0;
	w = (ufloat_t)0.0;
}

// o====================================================================================
// | Boundary conditions.
// o====================================================================================

template <typename ufloat_t>
__device__ __forceinline__
ufloat_t Cu_ImposeBC
(
	const int nbr_id, const ufloat_t f, const ufloat_t rho, const ufloat_t u, const ufloat_t v, const ufloat_t w, const ufloat_t x, const ufloat_t y, const ufloat_t z,
	const ufloat_t wp, const ufloat_t cxp, const ufloat_t cyp, const ufloat_t czp
)
{
	// --- HERE ---
	// Change the boundary conditions conditions here.
	// Uncomment blocks below for presets like lid-driven cavity and flow past cylinders.
	// --- HERE ---
	
	
	
	
	
	// LDC (2D).
// 	if (nbr_id == -4)
// 	{
// 		ufloat_t cdotu = (ufloat_t)(3.0)*( cxp*(ufloat_t)(0.05) );
// 		return f - (ufloat_t)(2.0)*wp*cdotu;
// 	}
	
	// FPSC (2D).
	if (nbr_id == -1 || nbr_id == -3 || nbr_id == -4)
	{
		ufloat_t cdotu = (ufloat_t)(3.0)*( cxp*(ufloat_t)(0.05) );
		return f - (ufloat_t)(2.0)*wp*cdotu;
	}
	if (nbr_id == -2)
	{
		ufloat_t cdotu = cxp*u + cyp*v + czp*w;
		cdotu = (ufloat_t)(1.0) + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*(u*u + v*v + w*w);
		return -f + (ufloat_t)(2.0)*wp*cdotu;
	}
	
	// LDC (3D).
// 	if (nbr_id == -6)
// 	{
// 		ufloat_t cdotu = (ufloat_t)(3.0)*( cxp*(ufloat_t)(0.05) );
// 		return f - (ufloat_t)(2.0)*wp*cdotu;
// 	}
	
	// FPSC (3D).
// 	if (nbr_id == -1 || nbr_id == -3 || nbr_id == -4 || nbr_id == -5 || nbr_id == -6)
// 	{
// 		ufloat_t cdotu = (ufloat_t)(3.0)*( cxp*(ufloat_t)(0.05) );
// 		return f - (ufloat_t)(2.0)*wp*cdotu;
// 	}
// 	if (nbr_id == -2)
// 	{
// 		ufloat_t cdotu = cxp*u + cyp*v + czp*w;
// 		cdotu = (ufloat_t)(1.0) + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*(u*u + v*v + w*w);
// 		return -f + (ufloat_t)(2.0)*wp*cdotu;
// 	}



	// If nothing defined, return a value of -1.
	return f;
}

#endif
