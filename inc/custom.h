#ifndef CUSTOM_H
#define CUSTOM_H

// o====================================================================================
// | Specified refinement region.
// o====================================================================================

template <typename ufloat_t>
__device__
bool Cu_RefineRegion
(
	int L, ufloat_t x, ufloat_t y, ufloat_t z, ufloat_t xp, ufloat_t yp, ufloat_t zp,
	int onb, int nbr_1, int nbr_2, int nbr_3, int nbr_4, int nbr_5, int nbr_6
)
{
	// --- HERE ---
	// Define a custom region of refinement here.
	// --- HERE ---
	
	return false;
}

// o====================================================================================
// | Initial conditions.
// o====================================================================================

template <typename ufloat_t>
__device__
void Cu_ComputeIC(ufloat_t &rho, ufloat_t &u, ufloat_t &v, ufloat_t &w, ufloat_t &x, ufloat_t &y, ufloat_t &z)
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
__device__
void Cu_ImposeBC
(
	int &nbr_id, ufloat_t &f, ufloat_t &rho, ufloat_t &u, ufloat_t &v, ufloat_t &w, ufloat_t &x, ufloat_t &y, ufloat_t &z,
	ufloat_t wp, ufloat_t cxp, ufloat_t cyp, ufloat_t czp, ufloat_t &cdotu
)
{
	// --- HERE ---
	// Change the boundary conditions conditions here.
	// Uncomment blocks below for presets like lid-driven cavity and flow past cylinders.
	// --- HERE ---
	
	
	
	
	
	// LDC (2D).
// 	if (nbr_id == -4)
// 	{
// 		cdotu = (ufloat_t)(3.0)*( cxp*(ufloat_t)(0.05) );
// 		f = f - (ufloat_t)(2.0)*wp*cdotu;
// 	}
	
	// FPSC (2D).
// 	if (nbr_id == -1 || nbr_id == -3 || nbr_id == -4)
// 	{
// 		cdotu = (ufloat_t)(3.0)*( cxp*(ufloat_t)(0.05) );
// 		f = f - (ufloat_t)(2.0)*wp*cdotu;
// 	}
// 	if (nbr_id == -2)
// 	{
// 		cdotu = cxp*u + cyp*v + czp*w;
// 		cdotu = (ufloat_t)(1.0) + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*(u*u + v*v + w*w);
// 		f = -f + (ufloat_t)(2.0)*wp*cdotu;
// 	}
	
	// LDC (3D).
// 	if (nbr_id == -6)
// 	{
// 		cdotu = (ufloat_t)(3.0)*( cxp*(ufloat_t)(0.05) );
// 		f = f - (ufloat_t)(2.0)*wp*cdotu;
// 	}
	
	// FPSC (3D).
	if (nbr_id == -1 || nbr_id == -3 || nbr_id == -4 || nbr_id == -5 || nbr_id == -6)
	{
		cdotu = (ufloat_t)(3.0)*( cxp*(ufloat_t)(0.05) );
		f = f - (ufloat_t)(2.0)*wp*cdotu;
	}
	if (nbr_id == -2)
	{
		cdotu = cxp*u + cyp*v + czp*w;
		cdotu = (ufloat_t)(1.0) + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*(u*u + v*v + w*w);
		f = -f + (ufloat_t)(2.0)*wp*cdotu;
	}
}

#endif
