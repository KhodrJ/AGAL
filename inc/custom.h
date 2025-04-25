#ifndef CUSTOM_H
#define CUSTOM_H

template <typename ufloat_t>
__device__
void Cu_ComputeIC(ufloat_t &rho, ufloat_t &u, ufloat_t &v, ufloat_t &w, ufloat_t &x, ufloat_t &y, ufloat_t &z)
{
	rho = (ufloat_t)1.0;
	u = (ufloat_t)0.1; //sin(M_PI*x);
	v = (ufloat_t)0.0; //sin(M_PI*y);
	w = (ufloat_t)0.0; //sin(M_PI*z);
}

template <typename ufloat_t>
__device__
void Cu_ImposeBC
(
	int &nbr_id, ufloat_t &f, ufloat_t &rho, ufloat_t &u, ufloat_t &v, ufloat_t &w, ufloat_t &x, ufloat_t &y, ufloat_t &z,
	ufloat_t wp, ufloat_t cxp, ufloat_t cyp, ufloat_t czp, ufloat_t &cdotu
)
{
	// LDC (2D).
// 	if (nbr_id == -4)
// 	{
// 		cdotu = (ufloat_t)(3.0)*( cxp*(ufloat_t)(0.05) );
// 		f = f - (ufloat_t)(2.0)*wp*cdotu;
// 	}
	
	
	
	// FPSC (2D).
	if (nbr_id == -1 || nbr_id == -3 || nbr_id == -4)
	{
		cdotu = (ufloat_t)(3.0)*( cxp*(ufloat_t)(0.1) );
		f = f - (ufloat_t)(2.0)*wp*cdotu;
	}
	if (nbr_id == -2)
	{
		cdotu = cxp*u + cyp*v + czp*w;
		cdotu = (ufloat_t)(1.0) + (ufloat_t)(4.5)*cdotu*cdotu - (ufloat_t)(1.5)*(u*u + v*v + w*w);
		f = -f + (ufloat_t)(2.0)*wp*cdotu;
	}
	
	
	
	// LDC (3D).
// 	if (nbr_id == -6)
// 	{
// 		cdotu = (ufloat_t)(3.0)*( cxp*(ufloat_t)(0.05) );
// 		f = f - (ufloat_t)(2.0)*wp*cdotu;
// 	}
	
	
	
	// FPSC (3D).
// 	if (nbr_id == -1 || nbr_id == -3 || nbr_id == -4 || nbr_id == -5 || nbr_id == -6)
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
}

#endif
