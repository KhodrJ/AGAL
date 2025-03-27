#ifndef CUSTOM_H
#define CUSTOM_H

template <typename ufloat_t>
__device__
void Cu_ComputeIC(ufloat_t &rho, ufloat_t &u, ufloat_t &v, ufloat_t &w, ufloat_t &x, ufloat_t &y, ufloat_t &z)
{
	rho = (ufloat_t)1.0;
	u = (ufloat_t)0.05; //sin(M_PI*x);
	v = (ufloat_t)0.0; //sin(M_PI*y);
	w = (ufloat_t)0.0; //sin(M_PI*z);
}

// template <typename ufloat_t>
// __device__
// void Cu_ImposeBC(int nbr_id, ufloat_t &f_p)
// {
// 	ufloat_t alpha = (ufloat_t)1.0;
// 	ufloat_t beta_1 = (ufloat_t)1.0;
// 	ufloat_t beta_2 = (ufloat_t)0.0;
// 	
// 	if (nbr_id == -1) {alpha=(ufloat_t)1.0; beta_1=(ufloat_t)1.0; beta_2=(ufloat_t)2.0;}
// 	
// 	// Assumes wall density unity, might generalize at some point.
// 	f_p = alpha*f_p + 2*w_p*(
// 		beta_1*() +
// 		beta_2*()
// 	);
// }

#endif
