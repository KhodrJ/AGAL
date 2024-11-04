/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#include "output.h"

#define Nbx 4





int PrintHelp()
{
	std::cout << "Usage:" << std::endl;
	std::cout << "-p:\n\tRender the processed frames.\n";
	std::cout << "-w:\n\tWrite the processed frames to a .vti file for later processing with Paraview.\n";
	std::cout << "-v:\n\tBe verbose with output.\n";
	std::cout << "--nsmooths=<int>:\n\tThe number of smoothing operations to perform for each frame. More smooths may improve renders.\n";
	std::cout << "--rparams <char params>:\n\tRender parameters. Char indicates the render type (default contour) and is followed by a set of parameters 'params' described as follows:\n\t+ c (contour): <string> (variable name for contour/isosurface) <int> (number of contours) <*d> (values of the contours)\n";
	
	return 0;
}

/*
888b     d888          888    888      
8888b   d8888          888    888      
88888b.d88888          888    888      
888Y88888P888  8888b.  888888 88888b.  
888 Y888P 888     "88b 888    888 "88b 
888  Y8P  888 .d888888 888    888  888 
888   "   888 888  888 Y88b.  888  888 
888       888 "Y888888  "Y888 888  888 
*/





struct complex_descending
{
	bool operator()(std::complex<float> cf1, std::complex<float> cf2)
	{
		if (std::real(cf1) == std::real(cf2))
			return std::real(cf1) > std::real(cf2);
		return std::real(cf1) > std::real(cf2);
	}
};

// Computes velocity gradient of 'data' array ('out' should be of length 9).
int VelGrad(int i, int j, int k, int p_var, int *Nxi_f, double dx, int vol, double *data, double *out, Eigen::Matrix3f *m_velgrad)
{
	// Definitions.
	double u = data[(i+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*k) + (p_var+0)*vol];
	double u_xM = 0.0;
	double u_xP = 0.0;
	double u_yM = 0.0;
	double u_yP = 0.0;
	double u_zM = 0.0;
	double u_zP = 0.0;
	double v = data[(i+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*k) + (p_var+1)*vol];
	double v_xM = 0.0;
	double v_xP = 0.0;
	double v_yM = 0.0;
	double v_yP = 0.0;
	double v_zM = 0.0;
	double v_zP = 0.0;
	double w = data[(i+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*k) + (p_var+2)*vol];
	double w_xM = 0.0;
	double w_xP = 0.0;
	double w_yM = 0.0;
	double w_yP = 0.0;
	double w_zM = 0.0;
	double w_zP = 0.0;
	
	// Corrections.
	if (i == 0)
	{
		u_xM = 2*u - data[((i+1)+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*k) + (p_var+0)*vol];
		v_xM = 2*v - data[((i+1)+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*k) + (p_var+1)*vol];
		w_xM = 2*w - data[((i+1)+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*k) + (p_var+2)*vol];
	}
	if (i > 0)
	{
		u_xM = data[((i-1)+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*k) + (p_var+0)*vol];
		v_xM = data[((i-1)+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*k) + (p_var+1)*vol];
		w_xM = data[((i-1)+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*k) + (p_var+2)*vol];
	}
	if (i == Nxi_f[0]-1)
	{
		u_xP = 2*u - data[((i-1)+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*k) + (p_var+0)*vol];
		v_xP = 2*v - data[((i-1)+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*k) + (p_var+1)*vol];
		w_xP = 2*w - data[((i-1)+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*k) + (p_var+2)*vol];
	}
	if (i < Nxi_f[0]-1)
	{
		u_xP = data[((i+1)+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*k) + (p_var+0)*vol];
		v_xP = data[((i+1)+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*k) + (p_var+1)*vol];
		w_xP = data[((i+1)+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*k) + (p_var+2)*vol];
	}
	if (j == 0)
	{
		u_yM = 2*u - data[(i+Nxi_f[0]*(j+1)+Nxi_f[0]*Nxi_f[1]*k) + (p_var+0)*vol];
		v_yM = 2*v - data[(i+Nxi_f[0]*(j+1)+Nxi_f[0]*Nxi_f[1]*k) + (p_var+1)*vol];
		w_yM = 2*w - data[(i+Nxi_f[0]*(j+1)+Nxi_f[0]*Nxi_f[1]*k) + (p_var+2)*vol];
	}
	if (j > 0)
	{
		u_yM = data[(i+Nxi_f[0]*(j-1)+Nxi_f[0]*Nxi_f[1]*k) + (p_var+0)*vol];
		v_yM = data[(i+Nxi_f[0]*(j-1)+Nxi_f[0]*Nxi_f[1]*k) + (p_var+1)*vol];
		w_yM = data[(i+Nxi_f[0]*(j-1)+Nxi_f[0]*Nxi_f[1]*k) + (p_var+2)*vol];
	}
	if (j == Nxi_f[1]-1)
	{
		u_yP = 2*u - data[(i+Nxi_f[0]*(j-1)+Nxi_f[0]*Nxi_f[1]*k) + (p_var+0)*vol];
		v_yP = 2*v - data[(i+Nxi_f[0]*(j-1)+Nxi_f[0]*Nxi_f[1]*k) + (p_var+1)*vol];
		w_yP = 2*w - data[(i+Nxi_f[0]*(j-1)+Nxi_f[0]*Nxi_f[1]*k) + (p_var+2)*vol];
	}
	if (j < Nxi_f[1]-1)
	{
		u_yP = data[(i+Nxi_f[0]*(j+1)+Nxi_f[0]*Nxi_f[1]*k) + (p_var+0)*vol];
		v_yP = data[(i+Nxi_f[0]*(j+1)+Nxi_f[0]*Nxi_f[1]*k) + (p_var+1)*vol];
		w_yP = data[(i+Nxi_f[0]*(j+1)+Nxi_f[0]*Nxi_f[1]*k) + (p_var+2)*vol];
	}
	if (Nxi_f[2]>1)
	{
		if (k == 0)
		{
			u_zM = 2*u - data[(i+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*(k+1)) + (p_var+0)*vol];
			v_zM = 2*v - data[(i+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*(k+1)) + (p_var+1)*vol];
			w_zM = 2*w - data[(i+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*(k+1)) + (p_var+2)*vol];
		}
		if (k > 0)
		{
			u_zM = data[(i+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*(k-1)) + (p_var+0)*vol];
			v_zM = data[(i+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*(k-1)) + (p_var+1)*vol];
			w_zM = data[(i+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*(k-1)) + (p_var+2)*vol];
		}
		if (k == Nxi_f[2]-1)
		{
			u_zP = 2*u - data[(i+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*(k-1)) + (p_var+0)*vol];
			v_zP = 2*v - data[(i+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*(k-1)) + (p_var+1)*vol];
			w_zP = 2*w - data[(i+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*(k-1)) + (p_var+2)*vol];
		}
		if (k < Nxi_f[2]-1)
		{
			u_zP = data[(i+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*(k+1)) + (p_var+0)*vol];
			v_zP = data[(i+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*(k+1)) + (p_var+1)*vol];
			w_zP = data[(i+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*(k+1)) + (p_var+2)*vol];
		}
	}
	
	// Calculations.
	// out[alpha + 3*beta] = del u_alpha / del x_beta
	out[0 + 3*0] = (u_xP - u_xM)/(2.0*dx);
	out[1 + 3*0] = (v_xP - v_xM)/(2.0*dx);
	out[2 + 3*0] = (w_xP - w_xM)/(2.0*dx);
	out[0 + 3*1] = (u_yP - u_yM)/(2.0*dx);
	out[1 + 3*1] = (v_yP - v_yM)/(2.0*dx);
	out[2 + 3*1] = (w_yP - w_yM)/(2.0*dx);
	out[0 + 3*2] = (u_zP - u_zM)/(2.0*dx);
	out[1 + 3*2] = (v_zP - v_zM)/(2.0*dx);
	out[2 + 3*2] = (w_zP - w_zM)/(2.0*dx);
	
	// Insert values into Eigen matrix.
	(*m_velgrad)(0,0) = out[0 + 3*0];
	(*m_velgrad)(1,0) = out[1 + 3*0];
	(*m_velgrad)(2,0) = out[2 + 3*0];
	(*m_velgrad)(0,1) = out[0 + 3*1];
	(*m_velgrad)(1,1) = out[1 + 3*1];
	(*m_velgrad)(2,1) = out[2 + 3*1];
	(*m_velgrad)(0,2) = out[0 + 3*2];
	(*m_velgrad)(1,2) = out[1 + 3*2];
	(*m_velgrad)(2,2) = out[2 + 3*2];
	
	return 0;
}

int ScalGrad(int i, int j, int k, int p_var, int *Nxi_f, double dx, int vol, double *data, double *out)
{
	// Definitions.
	double phi = data[(i+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*k) + p_var*vol];
	double phi_xM = 0.0;
	double phi_xP = 0.0;
	double phi_yM = 0.0;
	double phi_yP = 0.0;
	double phi_zM = 0.0;
	double phi_zP = 0.0;
	
	// Corrections.
	if (i == 0)
		phi_xM = 2*phi - data[((i+1)+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*k) + p_var*vol];
	if (i > 0)
		phi_xM = data[((i-1)+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*k) + p_var*vol];
	if (i == Nxi_f[0]-1)
		phi_xP = 2*phi - data[((i-1)+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*k) + p_var*vol];
	if (i < Nxi_f[0]-1)
		phi_xP = data[((i+1)+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*k) + p_var*vol];
	if (j == 0)
		phi_yM = 2*phi - data[(i+Nxi_f[0]*(j+1)+Nxi_f[0]*Nxi_f[1]*k) + p_var*vol];
	if (j > 0)
		phi_yM = data[(i+Nxi_f[0]*(j-1)+Nxi_f[0]*Nxi_f[1]*k) + p_var*vol];
	if (j == Nxi_f[1]-1)
		phi_yP = 2*phi - data[(i+Nxi_f[0]*(j-1)+Nxi_f[0]*Nxi_f[1]*k) + p_var*vol];
	if (j < Nxi_f[1]-1)
		phi_yP = data[(i+Nxi_f[0]*(j+1)+Nxi_f[0]*Nxi_f[1]*k) + p_var*vol];
	if (Nxi_f[2]>1)
	{
		if (k == 0)
			phi_zM = 2*phi - data[(i+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*(k+1)) + p_var*vol];
		if (k > 0)
			phi_zM = data[(i+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*(k-1)) + p_var*vol];
		if (k == Nxi_f[2]-1)
			phi_zP = 2*phi - data[(i+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*(k-1)) + p_var*vol];
		if (k < Nxi_f[2]-1)
			phi_zP = data[(i+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*(k+1)) + p_var*vol];
	}
	
	// Calculations.
	// out[alpha + 3*beta] = del u_alpha / del x_beta
	out[0] = (phi_xP - phi_xM)/(2.0*dx);
	out[1] = (phi_yP - phi_yM)/(2.0*dx);
	out[2] = (phi_zP - phi_zM)/(2.0*dx);
	
	return 0;
}

int CalculateYPlus(int i, int j, int k, int *Nxi_f, double dx, int vol, double *data)
{
	int kap = i + Nxi_f[0]*j + Nxi_f[0]*Nxi_f[1]*k;
	double odx = 1.0/dx;
	double yp_xm = 0.0;
	double yp_xp = 0.0;
	double yp_ym = 0.0;
	double yp_yp = 0.0;
	double yp_zm = 0.0;
	double yp_zp = 0.0;
	double tau_w = 0.0;
	double c1 = 0.0;
	double c2 = 0.0;
	
	if (Nxi_f[2]==1) // 2D
	{
		
	}
	else // 3D
	{
		// X
		if (i == 0)
		{
			c1 = odx*( (-71.0/24.0)*data[(i)+Nxi_f[0]*(j)+Nxi_f[0]*Nxi_f[1]*(k) + 2*vol] + 
				(141.0/24.0)*data[(i+1)+Nxi_f[0]*(j)+Nxi_f[0]*Nxi_f[1]*(k) + 2*vol] + 
				(-93.0/24.0)*data[(i+2)+Nxi_f[0]*(j)+Nxi_f[0]*Nxi_f[1]*(k) + 2*vol] + 
				(23.0/24.0)*data[(i+3)+Nxi_f[0]*(j)+Nxi_f[0]*Nxi_f[1]*(k) + 2*vol] );
			c2 = odx*( (-71.0/24.0)*data[(i)+Nxi_f[0]*(j)+Nxi_f[0]*Nxi_f[1]*(k) + 3*vol] + 
				(141.0/24.0)*data[(i+1)+Nxi_f[0]*(j)+Nxi_f[0]*Nxi_f[1]*(k) + 3*vol] + 
				(-93.0/24.0)*data[(i+2)+Nxi_f[0]*(j)+Nxi_f[0]*Nxi_f[1]*(k) + 3*vol] + 
				(23.0/24.0)*data[(i+3)+Nxi_f[0]*(j)+Nxi_f[0]*Nxi_f[1]*(k) + 3*vol] );
			tau_w = sqrt(c1*c1 + c2*c2);
			yp_xm = (dx/2.0)*sqrt(tau_w / (5.0e-6));
		}
		if (i == Nxi_f[0]-1)
		{
			c1 = odx*( (71.0/24.0)*data[(i)+Nxi_f[0]*(j)+Nxi_f[0]*Nxi_f[1]*(k) + 2*vol] + 
				(-141.0/24.0)*data[(i-1)+Nxi_f[0]*(j)+Nxi_f[0]*Nxi_f[1]*(k) + 2*vol] + 
				(93.0/24.0)*data[(i-2)+Nxi_f[0]*(j)+Nxi_f[0]*Nxi_f[1]*(k) + 2*vol] + 
				(-23.0/24.0)*data[(i-3)+Nxi_f[0]*(j)+Nxi_f[0]*Nxi_f[1]*(k) + 2*vol] );
			c2 = odx*( (71.0/24.0)*data[(i)+Nxi_f[0]*(j)+Nxi_f[0]*Nxi_f[1]*(k) + 3*vol] + 
				(-141.0/24.0)*data[(i-1)+Nxi_f[0]*(j)+Nxi_f[0]*Nxi_f[1]*(k) + 3*vol] + 
				(93.0/24.0)*data[(i-2)+Nxi_f[0]*(j)+Nxi_f[0]*Nxi_f[1]*(k) + 3*vol] + 
				(-23.0/24.0)*data[(i-3)+Nxi_f[0]*(j)+Nxi_f[0]*Nxi_f[1]*(k) + 3*vol] );
			tau_w = sqrt(c1*c1 + c2*c2);
			yp_xp = (dx/2.0)*sqrt(tau_w / (5.0e-6));
		}
		
		// Y
		if (j == 0)
		{
			c1 = odx*( (-71.0/24.0)*data[(i)+Nxi_f[0]*(j)+Nxi_f[0]*Nxi_f[1]*(k) + 1*vol] + 
				(141.0/24.0)*data[(i)+Nxi_f[0]*(j+1)+Nxi_f[0]*Nxi_f[1]*(k) + 1*vol] + 
				(-93.0/24.0)*data[(i)+Nxi_f[0]*(j+2)+Nxi_f[0]*Nxi_f[1]*(k) + 1*vol] + 
				(23.0/24.0)*data[(i)+Nxi_f[0]*(j+3)+Nxi_f[0]*Nxi_f[1]*(k) + 1*vol] );
			c2 = odx*( (-71.0/24.0)*data[(i)+Nxi_f[0]*(j)+Nxi_f[0]*Nxi_f[1]*(k) + 3*vol] + 
				(141.0/24.0)*data[(i)+Nxi_f[0]*(j+1)+Nxi_f[0]*Nxi_f[1]*(k) + 3*vol] + 
				(-93.0/24.0)*data[(i)+Nxi_f[0]*(j+2)+Nxi_f[0]*Nxi_f[1]*(k) + 3*vol] + 
				(23.0/24.0)*data[(i)+Nxi_f[0]*(j+3)+Nxi_f[0]*Nxi_f[1]*(k) + 3*vol] );
			tau_w = sqrt(c1*c1 + c2*c2);
			yp_ym = (dx/2.0)*sqrt(tau_w / (5.0e-6));
		}
		if (j == Nxi_f[1]-1)
		{
			c1 = odx*( (71.0/24.0)*data[(i)+Nxi_f[0]*(j)+Nxi_f[0]*Nxi_f[1]*(k) + 1*vol] + 
				(-141.0/24.0)*data[(i-1)+Nxi_f[0]*(j-1)+Nxi_f[0]*Nxi_f[1]*(k) + 1*vol] + 
				(93.0/24.0)*data[(i-2)+Nxi_f[0]*(j-2)+Nxi_f[0]*Nxi_f[1]*(k) + 1*vol] + 
				(-23.0/24.0)*data[(i-3)+Nxi_f[0]*(j-3)+Nxi_f[0]*Nxi_f[1]*(k) + 1*vol] );
			c2 = odx*( (71.0/24.0)*data[(i)+Nxi_f[0]*(j)+Nxi_f[0]*Nxi_f[1]*(k) + 3*vol] + 
				(-141.0/24.0)*data[(i-1)+Nxi_f[0]*(j-1)+Nxi_f[0]*Nxi_f[1]*(k) + 3*vol] + 
				(93.0/24.0)*data[(i-2)+Nxi_f[0]*(j-2)+Nxi_f[0]*Nxi_f[1]*(k) + 3*vol] + 
				(-23.0/24.0)*data[(i-3)+Nxi_f[0]*(j-3)+Nxi_f[0]*Nxi_f[1]*(k) + 3*vol] );
			tau_w = sqrt(c1*c1 + c2*c2);
			yp_yp = (dx/2.0)*sqrt(tau_w / (5.0e-6));
		}
		
		// Z
		if (k == 0)
		{
			c1 = odx*( (-71.0/24.0)*data[(i)+Nxi_f[0]*(j)+Nxi_f[0]*Nxi_f[1]*(k) + 1*vol] + 
				(141.0/24.0)*data[(i)+Nxi_f[0]*(j)+Nxi_f[0]*Nxi_f[1]*(k+1) + 1*vol] + 
				(-93.0/24.0)*data[(i)+Nxi_f[0]*(j)+Nxi_f[0]*Nxi_f[1]*(k+2) + 1*vol] + 
				(23.0/24.0)*data[(i)+Nxi_f[0]*(j)+Nxi_f[0]*Nxi_f[1]*(k+3) + 1*vol] );
			c2 = odx*( (-71.0/24.0)*data[(i)+Nxi_f[0]*(j)+Nxi_f[0]*Nxi_f[1]*(k) + 2*vol] + 
				(141.0/24.0)*data[(i)+Nxi_f[0]*(j)+Nxi_f[0]*Nxi_f[1]*(k+1) + 2*vol] + 
				(-93.0/24.0)*data[(i)+Nxi_f[0]*(j)+Nxi_f[0]*Nxi_f[1]*(k+2) + 2*vol] + 
				(23.0/24.0)*data[(i)+Nxi_f[0]*(j)+Nxi_f[0]*Nxi_f[1]*(k+3) + 2*vol] );
			tau_w = sqrt(c1*c1 + c2*c2);
			yp_zm = (dx/2.0)*sqrt(tau_w / (5.0e-6));
		}
		if (k == Nxi_f[2]-1)
		{
			c1 = odx*( (71.0/24.0)*data[(i)+Nxi_f[0]*(j)+Nxi_f[0]*Nxi_f[1]*(k) + 1*vol] + 
				(-141.0/24.0)*data[(i-1)+Nxi_f[0]*(j)+Nxi_f[0]*Nxi_f[1]*(k-1) + 1*vol] + 
				(93.0/24.0)*data[(i-2)+Nxi_f[0]*(j)+Nxi_f[0]*Nxi_f[1]*(k-2) + 1*vol] + 
				(-23.0/24.0)*data[(i-3)+Nxi_f[0]*(j)+Nxi_f[0]*Nxi_f[1]*(k-3) + 1*vol] );
			c2 = odx*( (71.0/24.0)*data[(i)+Nxi_f[0]*(j)+Nxi_f[0]*Nxi_f[1]*(k) + 2*vol] + 
				(-141.0/24.0)*data[(i-1)+Nxi_f[0]*(j)+Nxi_f[0]*Nxi_f[1]*(k-1) + 2*vol] + 
				(93.0/24.0)*data[(i-2)+Nxi_f[0]*(j)+Nxi_f[0]*Nxi_f[1]*(k-2) + 2*vol] + 
				(-23.0/24.0)*data[(i-3)+Nxi_f[0]*(j)+Nxi_f[0]*Nxi_f[1]*(k-3) + 2*vol] );
			tau_w = sqrt(c1*c1 + c2*c2);
			yp_zp = (dx/2.0)*sqrt(tau_w / (5.0e-6));
		}
	}
	
	data[kap + 12*vol] = std::max({yp_xm,yp_xp,yp_ym,yp_yp,yp_zm,yp_zp});

	return 0;
}





/*
8888888b.                                                     
888   Y88b                                                    
888    888                                                    
888   d88P 888d888 .d88b.   .d8888b .d88b.  .d8888b  .d8888b  
8888888P"  888P"  d88""88b d88P"   d8P  Y8b 88K      88K      
888        888    888  888 888     88888888 "Y8888b. "Y8888b. 
888        888    Y88..88P Y88b.   Y8b.          X88      X88 
888        888     "Y88P"   "Y8888P "Y8888   88888P'  88888P' 
*/
                                                              




int main(int argc, char *argv[])
{
	// o====================================================================================
	// | Define parameters.
	// o====================================================================================
	
	// From file.
	int           N_OUTPUT         = 0;
	int           P_OUTPUT         = 1;
	int           N_OUTPUT_START   = 0;
	int           N_PRECISION      = 1;
	int           N_DIM            = 3;
	int           N_LEVEL_START    = 0;
	int           N_PRINT_LEVELS   = 1;
	int           VOL_I_MIN        = 0;
	int           VOL_I_MAX        = 1;
	int           VOL_J_MIN        = 0;
	int           VOL_J_MAX        = 1;
	int           VOL_K_MIN        = 0;
	int           VOL_K_MAX        = 1;
	int           Nx               = -1;
	int           Ny               = -1;
	int           Nz               = -1;
	double        Lx               = 1.0;
	double        Ly               = 1.0;
	double        Lz               = 1.0;
	double        dx0              = -1.0;
	int           n_params         = 0;
	int           vol              = 1;
	int           vol_orig         = 1;
	double        vol_cell         = 0.0;
	double        ovol             = 0.0;
	long int      N_PROCS          = omp_get_max_threads();
	long int      buffer_length    = N_PROCS*1024*1024*1024;   // Default 1 GB buffer length for each process/thread.
	long int      buffer_req       = -1; // This will be the required buffer length determined by vol.
	
	// Program input (main).
	std::string   dirname          = "";
	std::string   filename         = "";
	bool          proceed_write    = false;
	bool          proceed_render   = false;
	bool          proceed_ave      = false;
	int           n_smooths        = 0;
	int           verbose          = false;
	
	// Program input (rendering).
	char          render_type      = 'c';
	std::string   c_var_name       = "Q-Criterion";
	int           c_var_count      = 1;
	double        *c_var_vals;
	
	
	// o====================================================================================
	// | Process program input.
	// o====================================================================================
	
	int i_nonopt = 0;
	if (argc == 2 && std::string(argv[1]) == "--help")
	{
		PrintHelp();
		
		return 0;
	}
	for (int i = 1; i < argc; i++)
	{
		bool short_opt = argv[i][0] == '-' && argv[i][1] != '-';
		bool long_opt = argv[i][0] == '-' && argv[i][1] == '-';
		
		// Short options.
		if (short_opt)
		{
			int j = 1;
			while (argv[i][j] != '\0')
			{
				switch (argv[i][j])
				{
					case 'p': proceed_render = true; std::cout << "[-] Proceeding with render...\n"; break;
					case 'w': proceed_write = true; std::cout << "[-] Proceeding with .vti write...\n"; break;
					case 'v': verbose = true; std::cout << "[-] Verbose output selected...\n"; break;
					case 'a': proceed_ave = true; std::cout << "[-] Time-averaging and turbulence selected...\n"; break;
					
					default: ;
				}
				
				j++;
			}
		}
		
		// Long options.
		if (long_opt)
		{
			bool processed_long_opt = false;
			
			if (std::string(&argv[i][2], 9) == "nsmooths=")
			{
				processed_long_opt = true;
				
				std::string s_nsmooths = std::string(&argv[i][11]);
				if (s_nsmooths != "")
					n_smooths = std::max( std::stoi( std::string(&argv[i][11]) ) , 0);
				else
					n_smooths = 0;
				std::cout << "[-] Using " << n_smooths << " smoothing operation(s) per frame...\n";
			}
			
			if (std::string(&argv[i][2], 7) == "rparams")
			{
				processed_long_opt = true;
				
				render_type = argv[i+1][0];
				if (render_type == 'c')
				{
					c_var_name = std::string(argv[i+2]);
					c_var_count = std::stoi( std::string(argv[i+3]) );
					c_var_vals = new double[c_var_count];
					
					for (int p = 0; p < c_var_count; p++)
						c_var_vals[p] = std::stod( std::string(argv[i+4+p]) );
					
					i += 4+c_var_count-1;
				}
			}
			
			
			if (!processed_long_opt)
			{
				std::string opt_name_full = std::string(&argv[i][2]);
				int equals_pos = opt_name_full.find('=');
				std::string opt_name = equals_pos==std::string::npos ? opt_name_full : std::string(opt_name_full.c_str(), opt_name_full.find('='));
				
				std::cout << "[-] Ignoring unknown/invalid option \"" << opt_name << "\"...\n";
				if (equals_pos==std::string::npos)
					std::cout << "    Looks like '=' operator is missing...\n";
			}
		}
		
		// Required.
		if (!short_opt && !long_opt)
		{
			// Set direcory.
			if (i_nonopt == 0)
			{
				dirname = argv[i];
				if (dirname[dirname.length()-1] != '/')
					dirname = dirname + std::string("/");
				filename = dirname + std::string("out_direct.dat");
				std::cout << "Using " << filename << std::endl;
			}
			
			i_nonopt++;
		}
		
	}
	if (i_nonopt == 0)
	{
		std::cout << "Please supply the input directory with the direct-output file..." << std::endl;
		return 1;
	}
	
	
	// o====================================================================================
	// | Reader metadata from file.
	// o====================================================================================
	
	std::ifstream output_file = std::ifstream(filename, std::ios::binary);
	if (!output_file)
	{
		std::cout << "Could not open file..." << std::endl;
		
		return 1;
	}
	char *buffer = new char[buffer_length];
	int init_read_length = (100)*sizeof(int)+(100)*sizeof(double);
	output_file.read(&buffer[0], init_read_length);
	long int pos = 0;
	memcpy(&N_OUTPUT, &buffer[pos], sizeof(int));         pos += sizeof(int);
	memcpy(&P_OUTPUT, &buffer[pos], sizeof(int));         pos += sizeof(int);
	memcpy(&N_OUTPUT_START, &buffer[pos], sizeof(int));   pos += sizeof(int);
	memcpy(&Nx, &buffer[pos], sizeof(int));               pos += sizeof(int);
	memcpy(&Ny, &buffer[pos], sizeof(int));               pos += sizeof(int);
	memcpy(&Nz, &buffer[pos], sizeof(int));               pos += sizeof(int);
	memcpy(&N_LEVEL_START, &buffer[pos], sizeof(int));    pos += sizeof(int);
	memcpy(&N_PRINT_LEVELS, &buffer[pos], sizeof(int));   pos += sizeof(int);
	memcpy(&N_PRECISION, &buffer[pos], sizeof(int));      pos += sizeof(int);
	memcpy(&VOL_I_MIN, &buffer[pos], sizeof(int));        pos += sizeof(int);
	memcpy(&VOL_I_MAX, &buffer[pos], sizeof(int));        pos += sizeof(int);
	memcpy(&VOL_J_MIN, &buffer[pos], sizeof(int));        pos += sizeof(int);
	memcpy(&VOL_J_MAX, &buffer[pos], sizeof(int));        pos += sizeof(int);
	memcpy(&VOL_K_MIN, &buffer[pos], sizeof(int));        pos += sizeof(int);
	memcpy(&VOL_K_MAX, &buffer[pos], sizeof(int));        pos += sizeof(int);
	pos = 100*sizeof(int);
	memcpy(&Lx, &buffer[pos], sizeof(double));            pos += sizeof(double);
	memcpy(&Ly, &buffer[pos], sizeof(double));            pos += sizeof(double);
	memcpy(&Lz, &buffer[pos], sizeof(double));            pos += sizeof(double);
	if (Nz == 1)
		N_DIM = 2;
	dx0 = Lx/(double)Nx;
	
	// Print recently read metadata.
	if (verbose)
	{
		std::cout << "[-] Loaded direct-output file with the following input:" << std::endl;
		std::cout << "    N_OUTPUT = " << N_OUTPUT << std::endl;
		std::cout << "    P_OUTPUT = " << P_OUTPUT << std::endl;
		std::cout << "    N_OUTPUT_START = " << N_OUTPUT_START << std::endl;
		std::cout << "    Nx = " << Nx << std::endl;
		std::cout << "    Ny = " << Ny << std::endl;
		std::cout << "    Nz = " << Nz << std::endl;
		std::cout << "    Lx = " << Lx << std::endl;
		std::cout << "    Ly = " << Ly << std::endl;
		std::cout << "    Lz = " << Lz << std::endl;
		std::cout << "    N_PRINT_LEVELS = " << N_PRINT_LEVELS << std::endl;
		std::cout << "    N_PRECISION = " << N_PRECISION << std::endl;
		std::cout << "    N_PROCS = " << N_PROCS << std::endl;
	}
	
	
	// Prepare resolution scaling parameters.
	int mult = 1;
	double dx_f = dx0;
	for (int L = 1; L < N_PRINT_LEVELS; L++) 
	{
		mult *= 2;
		dx_f *= 0.5;
	}
	int *mult_f = new int[N_PRINT_LEVELS];
	for (int L = 0; L < N_PRINT_LEVELS; L++)
		mult_f[L] = pow(2.0, (double)(N_PRINT_LEVELS-1-L));
	
	
	// Resolution array.
	int Nxi_f[3];
	Nxi_f[0] = (VOL_I_MAX-VOL_I_MIN)*Nbx;
	Nxi_f[1] = (VOL_J_MAX-VOL_J_MIN)*Nbx;
	Nxi_f[2] = (VOL_K_MAX-VOL_K_MIN)*Nbx; if (N_DIM==2) Nxi_f[2] = 1;
	vol_orig = Nxi_f[0]*Nxi_f[1]*Nxi_f[2];
	for (int d = 0; d < N_DIM; d++)
		Nxi_f[d] *= mult;
	vol = Nxi_f[0]*Nxi_f[1]*Nxi_f[2];
	vol_cell = dx_f*dx_f;
	if (N_DIM==3)
		vol_cell *= dx_f;
	ovol = 1.0/(double)vol;
	buffer_req = (3+1+2)*vol*sizeof(double) + sizeof(double);
	if (verbose)
		std::cout << "    Volume of frames: " << vol << std::endl;
	
	
	// Output files for volume-averaged density and total kinetic energy.
	double volave_density = 0.0;
	double volave_kinetic_energy = 0.0;
	std::ofstream output_volave = std::ofstream(dirname + std::string("volave.txt"));
	
	
	// Time-average collectors.
	//
	// Data (all time-averaged):
	// - Density [1:0]
	// - Velocity [3:1,2,3]
	// - Vorticity [3:4,5,6]
	// - Velocity Magnitude [1:7]
	// - Vorticity Magnitude [1:8]
	// - AMR Level [1:9]
	// - Block Id [1:10]
	// - Q-Criterion [1:11]
	// - Y+ [1:12]
	// - Pressure [1:13]
	//
	int n_var_ave = (1+3)+(3)+(1+1)+(1+1)+(1+1+1);
	double *tmp_data_ave[N_PROCS];
	double *tmp_data_rms[N_PROCS];
	for (int K = 0; K < N_PROCS; K++)
	{
		tmp_data_ave[K] = new double[n_var_ave*vol];
		tmp_data_rms[K] = new double[(1+3)*vol];
		for (int kap = 0; kap < n_var_ave*vol; kap++)
			tmp_data_ave[K][kap] = 0.0;
		for (int kap = 0; kap < (1+3)*vol; kap++)
			tmp_data_rms[K][kap] = 0.0;
	}
	
	
	// Turbulence data collectors.
	//
	// Data:
	// - Reynolds stress tensor [9:0-8]
	// - Mean velocity field [3:9-11]
	// - Mean pressure gradient vector [3:12,13,14]
	// - Turbulent kinetic energy [1:15]
	// - TKE production components [1:16]
	// - AMR Level [1:17]
	// - Block Id [1:18]
	//
	int n_data_turb = (9)+(3)+(3)+(1+1)+(2);
	double *tmp_data_rs[N_PROCS];
	for (int K = 0; K < N_PROCS; K++)
	{
		tmp_data_rs[K] = new double[n_data_turb*vol];
		for (int kap = 0; kap < n_data_turb*vol; kap++)
			tmp_data_rs[K][kap] = 0.0;
	}
	
	
	// o====================================================================================
	// | Start processing direct-output file.
	// o====================================================================================
	
	for (int K = 0; K < N_OUTPUT/N_PROCS+1; K++)
	{
		// Read density and velocity data for all threads.
		output_file.read(&buffer[0], N_PROCS*buffer_req);
		#pragma omp parallel
		{
		
		int t = omp_get_thread_num();
		int Kt = omp_get_thread_num() + K*N_PROCS;
		if (Kt < N_OUTPUT)
		{
			if (verbose)
				std::cout << std::endl << "PROCESSING FRAME No. " << Kt << std::endl;
			
			
			// Read frame data into arrays.
			// 
			// Data:
			// - Density [1:0] (read)
			// - Velocity [3:1,2,3] (read)
			// - Vorticity [3:4,5,6] (calc.)
			// - Velocity Magnitude [1:7] (calc.)
			// - Vorticity Magnitude [1:8] (calc.)
			// - AMR Level [1:9] (read)
			// - Block Id [1:10] (read)
			// - Q-Criterion [1:11] (calc.)
			// - Lambda-2 [1:12] (calc.)
			//
			int n_data = (1+3)+(3)+(1+1)+(1+1)+(1+1);
			double t_curr = -1;
			double *tmp_data = new double[n_data*vol];
			double *tmp_data_b = new double[n_data*vol];
			for (long int p = 0; p < n_data*vol; p++)
				tmp_data[p] = -1.0;
			
			// Memory copies.
			memcpy(&t_curr, &buffer[0 + t*buffer_req], sizeof(double));
			std::cout << "(t = " << t_curr << " s)" << std::endl;
			for (int d = 0; d < 3+1; d++)
				memcpy(&tmp_data[d*vol], &buffer[sizeof(double) + d*vol*sizeof(double) + t*buffer_req], vol*sizeof(double));
			memcpy(&tmp_data[9*vol], &buffer[sizeof(double) + 4*vol*sizeof(double) + t*buffer_req], vol*sizeof(double));
			memcpy(&tmp_data[10*vol], &buffer[sizeof(double) + 5*vol*sizeof(double) + t*buffer_req], vol*sizeof(double));
			
			
			// Update time-averages.
			for (int kap = 0; kap < vol; kap++)
			{
				for (int d = 0; d < 1+3; d++)
				{
					tmp_data_ave[t][kap+d*vol] = ((double)K*tmp_data_ave[t][kap+d*vol] + tmp_data[kap+d*vol]) / ( (double)(K+1) );
					tmp_data_rms[t][kap+d*vol] = ((double)K*tmp_data_rms[t][kap+d*vol] + tmp_data[kap+d*vol]*tmp_data[kap+d*vol]) / ( (double)(K+1) );
				}
			}
			
			
			// o====================================================================================
			// | Smoothing for better rendering.
			// o====================================================================================
			
			if (n_smooths > 0)
			{
				if (verbose)
					std::cout << "[-] Smoothing grid..." << std::endl;
			}
			for (int kap = 0; kap < vol; kap++)
			{
				for (int p = 0; p < N_DIM+1; p++)
					tmp_data_b[kap + p*vol] = tmp_data[kap + p*vol];
			}
			for (int l = 0; l < n_smooths; l++)
			{
				if (verbose)
					std::cout << "    Smoothing iteration " << l << "..." << std::endl;
				
				if (N_DIM == 2)
				{
					for (int j = 1; j < Nxi_f[1]-1; j++)
					{
						for (int i = 1; i < Nxi_f[0]-1; i++)
						{
							int kap = (i) + Nxi_f[0]*(j);
							bool proceed_smooth = true;
							if (tmp_data[(i+1) + Nxi_f[0]*(j) + 9*vol] < 0) proceed_smooth = false;
							if (tmp_data[(i-1) + Nxi_f[0]*(j) + 9*vol] < 0) proceed_smooth = false;
							if (tmp_data[(i) + Nxi_f[0]*(j+1) + 9*vol] < 0) proceed_smooth = false;
							if (tmp_data[(i) + Nxi_f[0]*(j-1) + 9*vol] < 0) proceed_smooth = false;
							
							if (proceed_smooth)
							{
								for (int p = 0; p < N_DIM+1; p++)
								{
									tmp_data_b[kap + p*vol] = (
										tmp_data[(i+1) + Nxi_f[0]*(j) + p*vol] +
										tmp_data[(i-1) + Nxi_f[0]*(j) + p*vol] +
										tmp_data[(i) + Nxi_f[0]*(j+1) + p*vol] +
										tmp_data[(i) + Nxi_f[0]*(j-1) + p*vol]
									)/4.0;
								}
							}
						}
					}
				}
				else
				{
					for (int k = 1; k < Nxi_f[2]-1; k++)
					{
						for (int j = 1; j < Nxi_f[1]-1; j++)
						{
							for (int i = 1; i < Nxi_f[0]-1; i++)
							{
								int kap = (i) + Nxi_f[0]*(j) + Nxi_f[0]*Nxi_f[1]*(k);
								bool proceed_smooth = true;
								if (tmp_data[(i+1) + Nxi_f[0]*(j) + Nxi_f[0]*Nxi_f[1]*(k) + 9*vol] < 0) proceed_smooth = false;
								if (tmp_data[(i-1) + Nxi_f[0]*(j) + Nxi_f[0]*Nxi_f[1]*(k) + 9*vol] < 0) proceed_smooth = false;
								if (tmp_data[(i) + Nxi_f[0]*(j+1) + Nxi_f[0]*Nxi_f[1]*(k) + 9*vol] < 0) proceed_smooth = false;
								if (tmp_data[(i) + Nxi_f[0]*(j-1) + Nxi_f[0]*Nxi_f[1]*(k) + 9*vol] < 0) proceed_smooth = false;
								if (tmp_data[(i) + Nxi_f[0]*(j) + Nxi_f[0]*Nxi_f[1]*(k+1) + 9*vol] < 0) proceed_smooth = false;
								if (tmp_data[(i) + Nxi_f[0]*(j) + Nxi_f[0]*Nxi_f[1]*(k-1) + 9*vol] < 0) proceed_smooth = false;
								
								if (proceed_smooth)
								{
									for (int p = 0; p < N_DIM+1; p++)
									{
										tmp_data_b[kap + p*vol] = (
											tmp_data[(i+1) + Nxi_f[0]*(j) + Nxi_f[0]*Nxi_f[1]*(k) + p*vol] +
											tmp_data[(i-1) + Nxi_f[0]*(j) + Nxi_f[0]*Nxi_f[1]*(k) + p*vol] +
											tmp_data[(i) + Nxi_f[0]*(j+1) + Nxi_f[0]*Nxi_f[1]*(k) + p*vol] +
											tmp_data[(i) + Nxi_f[0]*(j-1) + Nxi_f[0]*Nxi_f[1]*(k) + p*vol] +
											tmp_data[(i) + Nxi_f[0]*(j) + Nxi_f[0]*Nxi_f[1]*(k+1) + p*vol] +
											tmp_data[(i) + Nxi_f[0]*(j) + Nxi_f[0]*Nxi_f[1]*(k-1) + p*vol]
										)/6.0;
									}
								}
							}
						}
					}
				}
				
				for (int kap = 0; kap < vol; kap++)
				{
					for (int p = 0; p < N_DIM+1; p++)
						tmp_data[kap + p*vol] = tmp_data_b[kap + p*vol];
				}
			}
			if (n_smooths > 0)
			{
				if (verbose)
					std::cout << "    Finished smoothing grid..." << std::endl;
			}
			
			
			// o====================================================================================
			// | Compute remaining properties (e.g., vorticity, vector magnitudes...).
			// o====================================================================================
			
			if (verbose)
				std::cout << "[-] Computing properties..." << std::endl;
			volave_density = 0.0;
			volave_kinetic_energy = 0.0;
			for (int k = 0; k < Nxi_f[2]; k++)
			{
				for (int j = 0; j < Nxi_f[1]; j++)
				{
					for (int i = 0; i < Nxi_f[0]; i++)
					{
						long int kap = i + Nxi_f[0]*j + Nxi_f[0]*Nxi_f[1]*k;
						
						// Vorticity.
						double vel_grad[9];
						Eigen::Matrix3f m_velgrad(3,3);
						VelGrad(i,j,k,1,   Nxi_f,dx_f,vol,tmp_data,vel_grad,&m_velgrad);
						tmp_data[kap+4*vol] = vel_grad[2+3*1] - vel_grad[1+3*2];
						tmp_data[kap+5*vol] = vel_grad[0+3*2] - vel_grad[2+3*0];
						tmp_data[kap+6*vol] = vel_grad[1+3*0] - vel_grad[0+3*1];
						
						// Velocity and vorticity magnitudes.
						tmp_data[kap+7*vol] = sqrt(tmp_data[kap+1*vol]*tmp_data[kap+1*vol] + tmp_data[kap+2*vol]*tmp_data[kap+2*vol] + tmp_data[kap+3*vol]*tmp_data[kap+3*vol]);
						tmp_data[kap+8*vol] = sqrt(tmp_data[kap+4*vol]*tmp_data[kap+4*vol] + tmp_data[kap+5*vol]*tmp_data[kap+5*vol] + tmp_data[kap+6*vol]*tmp_data[kap+6*vol]);
						
						// Q- and Lambda2-criteria.
						tmp_data[kap+11*vol] = 
							vel_grad[0+3*0]*vel_grad[1+3*1] + vel_grad[1+3*1]*vel_grad[2+3*2] + vel_grad[2+3*2]*vel_grad[0+3*0] +
							-vel_grad[0+3*1]*vel_grad[1+3*0] - vel_grad[1+3*2]*vel_grad[2+3*1] - vel_grad[2+3*0]*vel_grad[0+3*2]
						;
// 						Eigen::Matrix3f m_S = 0.5f*(m_velgrad + m_velgrad.transpose());
// 						Eigen::Matrix3f m_O = 0.5f*(m_velgrad - m_velgrad.transpose());
// 						Eigen::Matrix3f m_A = m_S*m_S + m_O*m_O;
// 						Eigen::Vector3cf eigvals = m_A.eigenvalues();
// 						std::sort(eigvals.begin(), eigvals.end(), complex_descending());
// 						tmp_data[kap+12*vol] = (double)(std::real(eigvals(1)));
						tmp_data[kap+12*vol] = 0.0;
						
						
						
						
						
						// Copy AMR level and block Id to T-A and R-S array.
						if (Kt == 0)
						{
							tmp_data_ave[0][kap + 9*vol] = tmp_data[kap+9*vol];
							tmp_data_ave[0][kap + 10*vol] = tmp_data[kap+10*vol];
							tmp_data_rs[0][kap + 17*vol] = tmp_data[kap+9*vol];
							tmp_data_rs[0][kap + 18*vol] = tmp_data[kap+10*vol];
						}
						
						// Update volume-averaged data for this snapshot.
						volave_density = ((double)kap*volave_density + (tmp_data[kap+0*vol]-1.0)) / ( (double)(kap+1) );
						volave_kinetic_energy = ((double)kap*volave_kinetic_energy + 0.5*tmp_data[kap+7*vol]*tmp_data[kap+7*vol]) / ( (double)(kap+1) );
					}
				}
			}
			#pragma omp critical
			output_volave << t_curr << " " << volave_density << " " << volave_kinetic_energy << std::endl;
			if (verbose)
			{
				std::cout << "    Finished computing properties..." << std::endl;
				std::cout << std::endl;
			}
			
			
			// o====================================================================================
			// | VTK array setup if either rendering or writing selected.
			// o====================================================================================
			
			
			if (proceed_render || proceed_write)
			{
				// Define and fill VTK arrays.
					// Density.
				vtkNew<vtkDoubleArray> cell_data_density;
				cell_data_density->SetName("Density");
				cell_data_density->SetNumberOfComponents(1);
				cell_data_density->SetNumberOfTuples(vol);
					// Velocity.
				vtkNew<vtkDoubleArray> cell_data_velocity;
				cell_data_velocity->SetName("Velocity");
				cell_data_velocity->SetNumberOfComponents(3);
				cell_data_velocity->SetNumberOfTuples(vol);
					// Vorticity.
				vtkNew<vtkDoubleArray> cell_data_vorticity;
				cell_data_vorticity->SetName("Vorticity");
				cell_data_vorticity->SetNumberOfComponents(3);
				cell_data_vorticity->SetNumberOfTuples(vol);
					// Velocity Magnitude.
				vtkNew<vtkDoubleArray> cell_data_velmag;
				cell_data_velmag->SetName("Velocity Magnitude");
				cell_data_velmag->SetNumberOfComponents(1);
				cell_data_velmag->SetNumberOfTuples(vol);
					// Vorticity Magnitude.
				vtkNew<vtkDoubleArray> cell_data_vortmag;
				cell_data_vortmag->SetName("Vorticity Magnitude");
				cell_data_vortmag->SetNumberOfComponents(1);
				cell_data_vortmag->SetNumberOfTuples(vol);
					// Q-Criterion.
				vtkNew<vtkDoubleArray> cell_data_Q;
				cell_data_Q->SetName("Q-Criterion");
				cell_data_Q->SetNumberOfComponents(1);
				cell_data_Q->SetNumberOfTuples(vol);
					// Lambda2-Criterion.
				vtkNew<vtkDoubleArray> cell_data_L2;
				cell_data_L2->SetName("Lambda2-Criterion");
				cell_data_L2->SetNumberOfComponents(1);
				cell_data_L2->SetNumberOfTuples(vol);
					// AMR Level.
				vtkNew<vtkDoubleArray> cell_data_level;
				cell_data_level->SetName("AMR Level");
				cell_data_level->SetNumberOfComponents(1);
				cell_data_level->SetNumberOfTuples(vol);
					// Block Id.
				vtkNew<vtkDoubleArray> cell_data_blockid;
				cell_data_blockid->SetName("Block Id");
				cell_data_blockid->SetNumberOfComponents(1);
				cell_data_blockid->SetNumberOfTuples(vol);
				// |
				if (verbose)
					std::cout << "[-] Inserting data in VTK pointers..." << std::endl;
				for (long int kap = 0; kap < vol; kap++)
				{
					cell_data_density->SetTuple1(kap,
						tmp_data[kap+ 0*vol]
					);
					cell_data_velocity->SetTuple3(kap,
						tmp_data[kap+ 1*vol],
						tmp_data[kap+ 2*vol],
						tmp_data[kap+ 3*vol]
					);
					cell_data_vorticity->SetTuple3(kap, 
						tmp_data[kap+ 4*vol],
						tmp_data[kap+ 5*vol],
						tmp_data[kap+ 6*vol]
					);
					cell_data_velmag->SetTuple1(kap, 
						tmp_data[kap+ 7*vol]
					);
					cell_data_vortmag->SetTuple1(kap, 
						tmp_data[kap+ 8*vol]
					);
					cell_data_level->SetTuple1(kap,
						tmp_data[kap+ 9*vol]
					);
					cell_data_blockid->SetTuple1(kap,
						tmp_data[kap+ 10*vol]
					);
					cell_data_Q->SetTuple1(kap,
						tmp_data[kap+ 11*vol]
					);
					cell_data_L2->SetTuple1(kap,
						tmp_data[kap+ 12*vol]
					);
				}
				if (verbose)
					std::cout << "    Finished inserting data in VTK pointers..." << std::endl;
			
			
				// o====================================================================================
				// | Write to .vti file, if selected.
				// o====================================================================================
				
				
				if (verbose)
					std::cout << "[-] Creating uniform grid..." << std::endl;
					// Parameters and initialization.
				double origin[3] = {VOL_I_MIN*Nbx*dx0, VOL_J_MIN*Nbx*dx0, VOL_K_MIN*Nbx*dx0};
				double spacing[3] = {dx_f, dx_f, N_DIM==2?0.1*dx_f:dx_f};
				vtkNew<vtkUniformGrid> grid;
				vtkNew<vtkCellDataToPointData> cell_to_points;
				vtkNew<vtkContourFilter> contour;
					// Set up image data grid.
				grid->Initialize();
				grid->SetOrigin(origin);
				grid->SetSpacing(spacing);
				grid->SetDimensions(Nxi_f[0]+1, Nxi_f[1]+1, N_DIM==2?2:Nxi_f[2]+1);
				grid->GetCellData()->AddArray(cell_data_density);
				grid->GetCellData()->AddArray(cell_data_velocity);
				grid->GetCellData()->AddArray(cell_data_vorticity);
				grid->GetCellData()->AddArray(cell_data_velmag);
				grid->GetCellData()->AddArray(cell_data_vortmag);
				grid->GetCellData()->AddArray(cell_data_Q);
				grid->GetCellData()->AddArray(cell_data_L2);
				grid->GetCellData()->AddArray(cell_data_level);
				grid->GetCellData()->AddArray(cell_data_blockid);
					// Blank invalid cells (these are identified by negative AMR level).
				grid->AllocateCellGhostArray();
				vtkUnsignedCharArray *ghosts = grid->GetCellGhostArray();
				for (long int kap = 0; kap < vol; kap++)
				{
					if (tmp_data[kap + 9*vol] < 0)
						ghosts->SetValue(kap, ghosts->GetValue(kap) | vtkDataSetAttributes::HIDDENCELL);
				}
				if (verbose)
					std::cout << "    Finished creating uniform grid..." << std::endl;
				//|
				if (proceed_write)
				{
					if (verbose)
						std::cout << "Finished building VTK dataset, writing..." << std::endl;
					std::string file_name = dirname + std::string("out_") + std::to_string(Kt) + ".vti";
					vtkNew<vtkXMLImageDataWriter> writer;
					writer->SetInputData(grid);
					writer->SetFileName(file_name.c_str());
					writer->Write();
					if (verbose)
						std::cout << "[-] Finished writing VTK dataset..." << std::endl;
				}
			
			
				// o====================================================================================
				// | Render frame, if selected.
				// o====================================================================================
				
				if (proceed_render)
				{
					if (verbose)
						std::cout << "[-] Creating contours..." << std::endl;
					vtkNew<vtkRenderer> renderer;
					vtkNew<vtkDataSetMapper> ds_mapper;
					vtkNew<vtkPolyDataMapper> pd_mapper;
					vtkNew<vtkNamedColors> colors;
					vtkNew<vtkGraphicsFactory> graphics_factory;
					vtkNew<vtkLookupTable> lookup_table;
					vtkNew<vtkColorTransferFunction> transfer_function;
					vtkNew<vtkActor> actor;
					vtkNew<vtkRenderWindow> renderWindow;
					vtkNew<vtkCamera> camera;
					if (N_DIM == 2)
					{
						// Contour for velocity magnitude.
						//cell_to_points->GetImageDataOutput()->GetPointData()->SetActiveScalars("Vorticity Magnitude");
						
						
						// Setup offscreen rendering.
						if (verbose)
							std::cout << "[-] Setting up renderer..." << std::endl;;
						graphics_factory->SetOffScreenOnlyMode(1);
						graphics_factory->SetUseMesaClasses(1);
						//|
						transfer_function->AddRGBPoint(0, 0, 0, 0.5625);
						transfer_function->AddRGBPoint(0.0555555, 0, 0, 1);
						transfer_function->AddRGBPoint(0.18254, 0, 1, 1);
						transfer_function->AddRGBPoint(0.246032, 0.5, 1, 0.5);
						transfer_function->AddRGBPoint(0.309524, 1, 1, 0);
						transfer_function->AddRGBPoint(0.436508, 1, 0, 0);
						transfer_function->AddRGBPoint(0.5, 0.5, 0, 0);
						double tf_to_lut[256*3];
						transfer_function->GetTable(0.0,0.5,256,tf_to_lut);
						//|
						lookup_table->SetNumberOfTableValues(256);
						lookup_table->SetRange(0.0, 0.5);
						lookup_table->SetHueRange(0.667, 0.0);
						for (int p = 0; p < 256; p++)
							lookup_table->SetTableValue(p, tf_to_lut[0+p*3], tf_to_lut[1+p*3], tf_to_lut[2+p*3]);
						lookup_table->Build();
						//|
						grid->GetCellData()->SetActiveScalars("Vorticity Magnitude");

						
						ds_mapper->SetInputData(grid);
						ds_mapper->SetLookupTable(lookup_table);
						//
						actor->SetMapper(ds_mapper);
						
							// Create renderer.
						renderWindow->SetOffScreenRendering(1);
						renderWindow->AddRenderer(renderer);
							// Create camera.
						double cam_pos[3] = {0.5, 0.5, 2.5};
						double cam_view_up[3] = {0.0, 1.0, 0.0};
						double cam_focal_point[3] = {0.5, 0.5, 0.0};
						renderer->SetActiveCamera(camera);
						camera->SetPosition(cam_pos);
						camera->SetViewUp(cam_view_up);
						camera->SetFocalPoint(cam_focal_point);
							// Add actor to scene and render.
						renderer->AddActor(actor);
					}
					else
					{
							// Contours.
						cell_to_points->SetInputData(grid);
						cell_to_points->Update();
						cell_to_points->GetImageDataOutput()->GetPointData()->SetActiveScalars(c_var_name.c_str());
						contour->SetInputConnection(0, cell_to_points->GetOutputPort(0));
						contour->SetNumberOfContours(c_var_count);
						for (int p = 0; p < c_var_count; p++)
							contour->SetValue(0, c_var_vals[p]);
						if (verbose)
							std::cout << "    Finished creating contours..." << std::endl;
						
						if (verbose)
							std::cout << "[-] Setting up renderer..." << std::endl;
							// Setup offscreen rendering.
						graphics_factory->SetOffScreenOnlyMode(1);
						graphics_factory->SetUseMesaClasses(1);
							// Create mapper.
						//vtkNew<vtkPolyDataMapper> pd_mapper;
						pd_mapper->SetInputConnection(contour->GetOutputPort(0));
							// Create actor.
						actor->SetMapper(pd_mapper);
						actor->GetProperty()->SetColor(colors->GetColor3d("White").GetData());
							// Create renderer.
						renderWindow->SetOffScreenRendering(1);
						renderWindow->AddRenderer(renderer);
							// Create camera.
						double cam_pos[3] = {1.8, -2.5, 1.25};
						double cam_view_up[3] = {0.0, 0.0, 1.0};
						double cam_focal_point[3] = {0.5, 0.5, 0.5};
						//vtkNew<vtkCamera> camera;
						renderer->SetActiveCamera(camera);
						camera->SetPosition(cam_pos);
						camera->SetViewUp(cam_view_up);
						camera->SetFocalPoint(cam_focal_point);
							// Add actor to scene and render.
						renderer->AddActor(actor);
					}
					renderer->SetBackground(colors->GetColor3d("Black").GetData());
					if (verbose)
						std::cout << "    Finished setup, rendering..." << std::endl;
					renderWindow->SetSize(2048, 2048);
					renderWindow->Render();
					if (verbose)
						std::cout << "    Rendered, taking photo..." << std::endl;
						// Print to PNG.
					vtkNew<vtkWindowToImageFilter> windowToImageFilter;
					windowToImageFilter->SetInput(renderWindow);
					windowToImageFilter->Update();
					vtkNew<vtkPNGWriter> photographer;
					size_t n_zeros = 7;
					std::string iter_string = std::to_string(Kt);
					std::string padded_iter = std::string(n_zeros-std::min(n_zeros, iter_string.length()), '0') + iter_string;
					std::string photo_name = dirname + std::string("img/shot_") + padded_iter + ".png";
					photographer->SetFileName(photo_name.c_str());
					photographer->SetInputConnection(windowToImageFilter->GetOutputPort());
					photographer->Write();
					if (verbose)
						std::cout << "    Finished taking photo (no. " << Kt << ")..." << std::endl;
				}
			}
			
			
			// Free allocations.
			delete[] tmp_data;
			delete[] tmp_data_b;
			
			}
		}
	}
	
	// Close direct-output file.
	output_file.close();





/*
88888888888 d8b                                    d8888                                            d8b                   
    888     Y8P                                   d88888                                            Y8P                   
    888                                          d88P888                                                                  
    888     888 88888b.d88b.   .d88b.           d88P 888 888  888  .d88b.  888d888 8888b.   .d88b.  888 88888b.   .d88b.  
    888     888 888 "888 "88b d8P  Y8b         d88P  888 888  888 d8P  Y8b 888P"      "88b d88P"88b 888 888 "88b d88P"88b 
    888     888 888  888  888 88888888 888888 d88P   888 Y88  88P 88888888 888    .d888888 888  888 888 888  888 888  888 
    888     888 888  888  888 Y8b.           d8888888888  Y8bd8P  Y8b.     888    888  888 Y88b 888 888 888  888 Y88b 888 
    888     888 888  888  888  "Y8888       d88P     888   Y88P    "Y8888  888    "Y888888  "Y88888 888 888  888  "Y88888 
                                                                                                888                   888 
                                                                                           Y8b d88P              Y8b d88P 
                                                                                            "Y88P"                "Y88P"  
*/





	if (proceed_ave)
	{
		// Combine time-averaged datasets.
		for (int kap = 0; kap < (1+3)*vol; kap++)
		{
			tmp_data_ave[0][kap] = ((double)N_OUTPUT/(double)N_PROCS)*tmp_data_ave[0][kap];
			tmp_data_rms[0][kap] = ((double)N_OUTPUT/(double)N_PROCS)*tmp_data_rms[0][kap];
		}
		for (int K = 1; K < N_PROCS; K++)
		{
			// Final values stored in first array, then normalized afterward.
			for (int kap = 0; kap < (1+3)*vol; kap++)
			{
				tmp_data_ave[0][kap] += ((double)N_OUTPUT/(double)N_PROCS)*tmp_data_ave[K][kap];
				tmp_data_rms[0][kap] += ((double)N_OUTPUT/(double)N_PROCS)*tmp_data_rms[K][kap];
			}
		}
		for (int kap = 0; kap < (1+3)*vol; kap++)
		{
			tmp_data_ave[0][kap] /= (double)N_OUTPUT;
			tmp_data_rms[0][kap] = sqrt( tmp_data_rms[0][kap] / (double)N_OUTPUT );
		}
		
		
		// Compute remaining properties (e.g., vorticity, vector magnitudes...).
		if (verbose)
			std::cout << "[-] Computing properties..." << std::endl;
		for (int k = 0; k < Nxi_f[2]; k++)
		{
			for (int j = 0; j < Nxi_f[1]; j++)
			{
				for (int i = 0; i < Nxi_f[0]; i++)
				{
					long int kap = i + Nxi_f[0]*j + Nxi_f[0]*Nxi_f[1]*k;
					
					// Vorticity.
					double vel_grad[9];
					Eigen::Matrix3f m_velgrad(3,3);
					VelGrad(i,j,k,1,   Nxi_f,dx_f,vol,tmp_data_ave[0],vel_grad,&m_velgrad);
					tmp_data_ave[0][kap+4*vol] = vel_grad[2+3*1] - vel_grad[1+3*2];
					tmp_data_ave[0][kap+5*vol] = vel_grad[0+3*2] - vel_grad[2+3*0];
					tmp_data_ave[0][kap+6*vol] = vel_grad[1+3*0] - vel_grad[0+3*1];
					
					// Velocity and vorticity magnitudes.
					tmp_data_ave[0][kap+7*vol] = sqrt(tmp_data_ave[0][kap+1*vol]*tmp_data_ave[0][kap+1*vol] + tmp_data_ave[0][kap+2*vol]*tmp_data_ave[0][kap+2*vol] + tmp_data_ave[0][kap+3*vol]*tmp_data_ave[0][kap+3*vol]);
					tmp_data_ave[0][kap+8*vol] = sqrt(tmp_data_ave[0][kap+4*vol]*tmp_data_ave[0][kap+4*vol] + tmp_data_ave[0][kap+5*vol]*tmp_data_ave[0][kap+5*vol] + tmp_data_ave[0][kap+6*vol]*tmp_data_ave[0][kap+6*vol]);
					
					
					// Q-criterion.
					tmp_data_ave[0][kap+11*vol] = 
						vel_grad[0+3*0]*vel_grad[1+3*1] + vel_grad[1+3*1]*vel_grad[2+3*2] + vel_grad[2+3*2]*vel_grad[0+3*0] +
						-vel_grad[0+3*1]*vel_grad[1+3*0] - vel_grad[1+3*2]*vel_grad[2+3*1] - vel_grad[2+3*0]*vel_grad[0+3*2]
					;
					
					// Y+.
					CalculateYPlus(i,j,k, Nxi_f,dx_f,vol,tmp_data_ave[0]);
					
					// Pressure.
					tmp_data_ave[0][kap+13*vol] = (tmp_data_ave[0][kap+0*vol]-1.0)/3.0;
				}
			}
		}
		if (verbose)
		{
			std::cout << "    Finished computing properties..." << std::endl;
			std::cout << std::endl;
		}
		
		
		// Fill VTK arrays.
			// Density.
		vtkNew<vtkDoubleArray> cell_data_density;
		cell_data_density->SetName("TA Density");
		cell_data_density->SetNumberOfComponents(1);
		cell_data_density->SetNumberOfTuples(vol);
			// Velocity.
		vtkNew<vtkDoubleArray> cell_data_velocity;
		cell_data_velocity->SetName("TA Velocity");
		cell_data_velocity->SetNumberOfComponents(3);
		cell_data_velocity->SetNumberOfTuples(vol);
			// Velocity.
		vtkNew<vtkDoubleArray> cell_data_velrms;
		cell_data_velrms->SetName("RMS Velocity");
		cell_data_velrms->SetNumberOfComponents(3);
		cell_data_velrms->SetNumberOfTuples(vol);
			// Vorticity.
		vtkNew<vtkDoubleArray> cell_data_vorticity;
		cell_data_vorticity->SetName("TA Vorticity");
		cell_data_vorticity->SetNumberOfComponents(3);
		cell_data_vorticity->SetNumberOfTuples(vol);
			// Velocity Magnitude.
		vtkNew<vtkDoubleArray> cell_data_velmag;
		cell_data_velmag->SetName("TA Velocity Magnitude");
		cell_data_velmag->SetNumberOfComponents(1);
		cell_data_velmag->SetNumberOfTuples(vol);
			// Vorticity Magnitude.
		vtkNew<vtkDoubleArray> cell_data_vortmag;
		cell_data_vortmag->SetName("TA Vorticity Magnitude");
		cell_data_vortmag->SetNumberOfComponents(1);
		cell_data_vortmag->SetNumberOfTuples(vol);
			// Q-Criterion.
		vtkNew<vtkDoubleArray> cell_data_Q;
		cell_data_Q->SetName("TA Q-Criterion");
		cell_data_Q->SetNumberOfComponents(1);
		cell_data_Q->SetNumberOfTuples(vol);
			// AMR Level.
		vtkNew<vtkDoubleArray> cell_data_level;
		cell_data_level->SetName("AMR Level");
		cell_data_level->SetNumberOfComponents(1);
		cell_data_level->SetNumberOfTuples(vol);
			// Block Id.
		vtkNew<vtkDoubleArray> cell_data_blockid;
		cell_data_blockid->SetName("Block Id");
		cell_data_blockid->SetNumberOfComponents(1);
		cell_data_blockid->SetNumberOfTuples(vol);
			// Y+.
		vtkNew<vtkDoubleArray> cell_data_yp;
		cell_data_yp->SetName("Y+");
		cell_data_yp->SetNumberOfComponents(1);
		cell_data_yp->SetNumberOfTuples(vol);
			// Pressure.
		vtkNew<vtkDoubleArray> cell_data_pressure;
		cell_data_pressure->SetName("Pressure");
		cell_data_pressure->SetNumberOfComponents(1);
		cell_data_pressure->SetNumberOfTuples(vol);
		// |
		if (verbose)
			std::cout << "[-] Inserting data in VTK pointers..." << std::endl;
		for (long int kap = 0; kap < vol; kap++)
		{
			cell_data_density->SetTuple1(kap,
				tmp_data_ave[0][kap+ 0*vol]
			);
			cell_data_velocity->SetTuple3(kap,
				tmp_data_ave[0][kap+ 1*vol],
				tmp_data_ave[0][kap+ 2*vol],
				tmp_data_ave[0][kap+ 3*vol]
			);
			cell_data_velrms->SetTuple3(kap,
				tmp_data_rms[0][kap+ 1*vol],
				tmp_data_rms[0][kap+ 2*vol],
				tmp_data_rms[0][kap+ 3*vol]
			);
			cell_data_vorticity->SetTuple3(kap, 
				tmp_data_ave[0][kap+ 4*vol],
				tmp_data_ave[0][kap+ 5*vol],
				tmp_data_ave[0][kap+ 6*vol]
			);
			cell_data_velmag->SetTuple1(kap, 
				tmp_data_ave[0][kap+ 7*vol]
			);
			cell_data_vortmag->SetTuple1(kap, 
				tmp_data_ave[0][kap+ 8*vol]
			);
			cell_data_level->SetTuple1(kap,
				tmp_data_ave[0][kap+ 9*vol]
			);
			cell_data_blockid->SetTuple1(kap,
				tmp_data_ave[0][kap+ 10*vol]
			);
			cell_data_Q->SetTuple1(kap,
				tmp_data_ave[0][kap+ 11*vol]
			);
			cell_data_yp->SetTuple1(kap,
				tmp_data_ave[0][kap+ 12*vol]
			);
			cell_data_pressure->SetTuple1(kap,
				tmp_data_ave[0][kap+ 13*vol]
			);
		}
		if (verbose)
			std::cout << "    Finished inserting data in VTK pointers..." << std::endl;
		
		
		// Printing.
		if (verbose)
			std::cout << "[-] Creating uniform grid..." << std::endl;
			// Parameters and initialization.
		double origin[3] = {VOL_I_MIN*Nbx*dx0, VOL_J_MIN*Nbx*dx0, VOL_K_MIN*Nbx*dx0};
		double spacing[3] = {dx_f, dx_f, dx_f};
		vtkNew<vtkUniformGrid> grid;
		vtkNew<vtkCellDataToPointData> cell_to_points;
		vtkNew<vtkContourFilter> contour;
			// Set up image data grid.
		grid->Initialize();
		grid->SetOrigin(origin);
		grid->SetSpacing(spacing);
		grid->SetDimensions(Nxi_f[0]+1, Nxi_f[1]+1, N_DIM==2?2:Nxi_f[2]+1);
		grid->GetCellData()->AddArray(cell_data_density);
		grid->GetCellData()->AddArray(cell_data_velocity);
		grid->GetCellData()->AddArray(cell_data_velrms);
		grid->GetCellData()->AddArray(cell_data_vorticity);
		grid->GetCellData()->AddArray(cell_data_velmag);
		grid->GetCellData()->AddArray(cell_data_vortmag);
		grid->GetCellData()->AddArray(cell_data_Q);
		grid->GetCellData()->AddArray(cell_data_yp);
		grid->GetCellData()->AddArray(cell_data_pressure);
		grid->GetCellData()->AddArray(cell_data_level);
		grid->GetCellData()->AddArray(cell_data_blockid);
			// Blank invalid cells (these are identified by negative AMR level).
		grid->AllocateCellGhostArray();
		vtkUnsignedCharArray *ghosts = grid->GetCellGhostArray();
		for (long int kap = 0; kap < vol; kap++)
		{
			if (tmp_data_ave[0][kap + 9*vol] < 0)
				ghosts->SetValue(kap, ghosts->GetValue(kap) | vtkDataSetAttributes::HIDDENCELL);
		}
		if (verbose)
			std::cout << "    Finished creating uniform grid..." << std::endl;
		//|
		if (verbose)
			std::cout << "Finished building VTK dataset, writing..." << std::endl;
		std::string file_name = dirname + std::string("ta_out.vti");
		vtkNew<vtkXMLImageDataWriter> writer;
		writer->SetInputData(grid);
		writer->SetFileName(file_name.c_str());
		writer->Write();
		if (verbose)
			std::cout << "Finished writing VTK dataset..." << std::endl;
		
		
		// Print to a binary file for Reynolds stress later.
		
		
		// Extract mean profiles across cavity.
		std::ofstream output_ta_profiles = std::ofstream(dirname + std::string("ta_out_profiles.txt"));
		std::ofstream output_rms_profiles = std::ofstream(dirname + std::string("ta_out_profiles_rms.txt"));
		if (N_DIM==3)
		{
			output_ta_profiles << "u z x w" << std::endl;
			output_rms_profiles << "u z x w" << std::endl;
			
			int i_half = Nxi_f[0]/2-1;
			int j_half = Nxi_f[1]/2-1;
			int k_half = Nxi_f[2]/2-1;
			double *ta_vertical_profile = new double[Nxi_f[2]];
			double *ta_horizontal_profile = new double[Nxi_f[0]];
			double *rms_vertical_profile = new double[Nxi_f[2]];
			double *rms_horizontal_profile = new double[Nxi_f[0]];
			
			for (int k = 0; k < Nxi_f[2]; k++)
			{
				int kap_00 = (i_half+0) + Nxi_f[0]*(j_half+0) + Nxi_f[0]*Nxi_f[1]*k;
				int kap_01 = (i_half+0) + Nxi_f[0]*(j_half+1) + Nxi_f[0]*Nxi_f[1]*k;
				int kap_10 = (i_half+1) + Nxi_f[0]*(j_half+0) + Nxi_f[0]*Nxi_f[1]*k;
				int kap_11 = (i_half+1) + Nxi_f[0]*(j_half+1) + Nxi_f[0]*Nxi_f[1]*k;
				
				ta_vertical_profile[k] = 0.25*(
					tmp_data_ave[0][kap_00+1*vol]+
					tmp_data_ave[0][kap_01+1*vol]+
					tmp_data_ave[0][kap_10+1*vol]+
					tmp_data_ave[0][kap_11+1*vol]
				);
				rms_vertical_profile[k] = 0.25*(
					tmp_data_rms[0][kap_00+1*vol]+
					tmp_data_rms[0][kap_01+1*vol]+
					tmp_data_rms[0][kap_10+1*vol]+
					tmp_data_rms[0][kap_11+1*vol]
				);
			}
			
			for (int i = 0; i < Nxi_f[0]; i++)
			{
				int kap_00 = i + Nxi_f[0]*(j_half+0) + Nxi_f[0]*Nxi_f[1]*(k_half+0);
				int kap_01 = i + Nxi_f[0]*(j_half+0) + Nxi_f[0]*Nxi_f[1]*(k_half+1);
				int kap_10 = i + Nxi_f[0]*(j_half+1) + Nxi_f[0]*Nxi_f[1]*(k_half+0);
				int kap_11 = i + Nxi_f[0]*(j_half+1) + Nxi_f[0]*Nxi_f[1]*(k_half+1);
				
				ta_horizontal_profile[i] = 0.25*(
					tmp_data_ave[0][kap_00+3*vol]+
					tmp_data_ave[0][kap_01+3*vol]+
					tmp_data_ave[0][kap_10+3*vol]+
					tmp_data_ave[0][kap_11+3*vol]
				);
				rms_horizontal_profile[i] = 0.25*(
					tmp_data_rms[0][kap_00+3*vol]+
					tmp_data_rms[0][kap_01+3*vol]+
					tmp_data_rms[0][kap_10+3*vol]+
					tmp_data_rms[0][kap_11+3*vol]
				);
			}
			
			for (int l = 0; l < std::max(Nxi_f[0],Nxi_f[2]); l++)
			{
				if (l < Nxi_f[2])
					output_ta_profiles << ta_vertical_profile[l] << " " << dx_f*(l+0.5) << " ";
				else
					output_ta_profiles << "- - ";
				
				if (l < Nxi_f[0])
					output_ta_profiles << dx_f*(l+0.5) << " " << ta_horizontal_profile[l] << std::endl;
				else
					output_ta_profiles << "- -" << std::endl;
			}
			for (int l = 0; l < std::max(Nxi_f[0],Nxi_f[2]); l++)
			{
				if (l < Nxi_f[2])
					output_rms_profiles << rms_vertical_profile[l] << " " << dx_f*(l+0.5) << " ";
				else
					output_rms_profiles << "- - ";
				
				if (l < Nxi_f[0])
					output_rms_profiles << dx_f*(l+0.5) << " " << rms_horizontal_profile[l] << std::endl;
				else
					output_rms_profiles << "- -" << std::endl;
			}
			
			delete[] ta_vertical_profile;
			delete[] ta_horizontal_profile;
			delete[] rms_vertical_profile;
			delete[] rms_horizontal_profile;
		}
		output_ta_profiles.close();
		output_rms_profiles.close();
	}





/*
88888888888               888               888                                    
    888                   888               888                                    
    888                   888               888                                    
    888  888  888 888d888 88888b.  888  888 888  .d88b.  88888b.   .d8888b .d88b.  
    888  888  888 888P"   888 "88b 888  888 888 d8P  Y8b 888 "88b d88P"   d8P  Y8b 
    888  888  888 888     888  888 888  888 888 88888888 888  888 888     88888888 
    888  Y88b 888 888     888 d88P Y88b 888 888 Y8b.     888  888 Y88b.   Y8b.     
    888   "Y88888 888     88888P"   "Y88888 888  "Y8888  888  888  "Y8888P "Y8888  
*/





	// o====================================================================================
	// | Reread direct-output file and obtain turbulence data using means.
	// o====================================================================================
	
	if (proceed_ave)
	{
		// Re-open direct-output file.
		output_file.open(filename, std::ios::binary);
		output_file.read(&buffer[0], init_read_length); // Ignore metadata this time, already processed.
		
		
		// Output files for probed velocity fluctuations.
		std::ofstream output_velfluc_UZ = std::ofstream(dirname + std::string("velfluc_UZ.txt"));
		std::ofstream output_velfluc_VX = std::ofstream(dirname + std::string("velfluc_VX.txt"));
		
		
		// Loop over temporal dataset again, this time extracting Reynolds stresses.
		for (int K = 0; K < N_OUTPUT/N_PROCS+1; K++)
		{
			// Read density and velocity data for all threads.
			output_file.read(&buffer[0], N_PROCS*buffer_req);
			#pragma omp parallel
			{
			
			int t = omp_get_thread_num();
			int Kt = omp_get_thread_num() + K*N_PROCS;
			if (Kt < N_OUTPUT)
			{
				if (verbose)
					std::cout << std::endl << "PROCESSING FRAME No. " << Kt << std::endl;
				
				
				// Read frame data into arrays.
				// 
				// Data:
				// - Density [1:0] (read)
				// - Velocity flucatation [3:1,2,3] (read)
				// - AMR Level [1:4] (read)
				// - Block Id [1:5] (read)
				//
				int n_data = (1+3)+(1+1);
				double t_curr = -1;
				double *tmp_data = new double[n_data*vol];
				for (long int p = 0; p < n_data*vol; p++)
					tmp_data[p] = -1.0;
				
				// Memory copies.
				memcpy(&t_curr, &buffer[0 + t*buffer_req], sizeof(double));
				if (verbose)
					std::cout << "(t = " << t_curr << " s)" << std::endl;
				for (int d = 0; d < 3+1; d++)
					memcpy(&tmp_data[d*vol], &buffer[sizeof(double) + d*vol*sizeof(double) + t*buffer_req], vol*sizeof(double));
				memcpy(&tmp_data[4*vol], &buffer[sizeof(double) + 4*vol*sizeof(double) + t*buffer_req], vol*sizeof(double));
				memcpy(&tmp_data[5*vol], &buffer[sizeof(double) + 5*vol*sizeof(double) + t*buffer_req], vol*sizeof(double));
				
				
				// Replace total velocity with velocity fluctuations.
				for (int kap = 0; kap < vol; kap++)
				{
					for (int d = 0; d < 1+3; d++)
						tmp_data[kap+d*vol] = tmp_data[kap+d*vol] - tmp_data_ave[0][kap+d*vol];
				}
				
				
				// Record velocity fluctations in some parts of the domain.
				#pragma omp critical
				{
					if (Kt == 0)
					{
						output_velfluc_UZ << -1 << " ";
						for (int k = 0; k < Nxi_f[2]; k++)
						{
							int j = Nxi_f[1]/2;
							//for (int j = 0; j < Nxi_f[1]; j+=16*mult)
							{
								int i = Nxi_f[0]/2;
								//for (int i = 0; i < Nxi_f[0]; i+=16*mult)
								{
									long int kap = i + Nxi_f[0]*j + Nxi_f[0]*Nxi_f[1]*k;
									double x_ijk = dx_f/2.0 + i*dx_f;
									double y_ijk = dx_f/2.0 + j*dx_f;
									double z_ijk = dx_f/2.0 + k*dx_f;
									
									// Format: tn x1,y1,z1, rho',u',v',w', x2,y2,z2,...
									output_velfluc_UZ << x_ijk << " " << y_ijk << " " << z_ijk << " " << tmp_data_ave[0][kap+0*vol] << " " << tmp_data_ave[0][kap+1*vol] << " " << tmp_data_ave[0][kap+2*vol] << " " << tmp_data_ave[0][kap+3*vol] << " ";
								}
							}
						}
						output_velfluc_UZ << std::endl;
						output_velfluc_VX << Kt << " ";
						int k = Nxi_f[2]/2;
						//for (int k = 0; k < Nxi_f[2]; k++)
						{
							int j = Nxi_f[1]/2;
							//for (int j = 0; j < Nxi_f[1]; j+=16*mult)
							{
								for (int i = 0; i < Nxi_f[0]; i++)
								{
									long int kap = i + Nxi_f[0]*j + Nxi_f[0]*Nxi_f[1]*k;
									double x_ijk = dx_f/2.0 + i*dx_f;
									double y_ijk = dx_f/2.0 + j*dx_f;
									double z_ijk = dx_f/2.0 + k*dx_f;
									
									// Format: tn x1,y1,z1, u',v',w', x2,y2,z2,...
									output_velfluc_VX << x_ijk << " " << y_ijk << " " << z_ijk << " " << tmp_data_ave[0][kap+0*vol] << " " << tmp_data_ave[0][kap+1*vol] << " " << tmp_data_ave[0][kap+2*vol] << " " << tmp_data_ave[0][kap+3*vol] << " ";
								}
							}
						}
						output_velfluc_VX << std::endl;
					}
					output_velfluc_UZ << Kt << " ";
					for (int k = 0; k < Nxi_f[2]; k++)
					{
						int j = Nxi_f[1]/2;
						//for (int j = 0; j < Nxi_f[1]; j+=16*mult)
						{
							int i = Nxi_f[0]/2;
							//for (int i = 0; i < Nxi_f[0]; i+=16*mult)
							{
								long int kap = i + Nxi_f[0]*j + Nxi_f[0]*Nxi_f[1]*k;
								double x_ijk = dx_f/2.0 + i*dx_f;
								double y_ijk = dx_f/2.0 + j*dx_f;
								double z_ijk = dx_f/2.0 + k*dx_f;
								
								// Format: tn x1,y1,z1, u',v',w', x2,y2,z2,...
								output_velfluc_UZ << x_ijk << " " << y_ijk << " " << z_ijk << " " << tmp_data[kap+0*vol] << " " << tmp_data[kap+1*vol] << " " << tmp_data[kap+2*vol] << " " << tmp_data[kap+3*vol] << " ";
							}
						}
					}
					output_velfluc_UZ << std::endl;
					output_velfluc_VX << Kt << " ";
					int k = Nxi_f[2]/2;
					//for (int k = 0; k < Nxi_f[2]; k++)
					{
						int j = Nxi_f[1]/2;
						//for (int j = 0; j < Nxi_f[1]; j+=16*mult)
						{
							for (int i = 0; i < Nxi_f[0]; i++)
							{
								long int kap = i + Nxi_f[0]*j + Nxi_f[0]*Nxi_f[1]*k;
								double x_ijk = dx_f/2.0 + i*dx_f;
								double y_ijk = dx_f/2.0 + j*dx_f;
								double z_ijk = dx_f/2.0 + k*dx_f;
								
								// Format: tn x1,y1,z1, u',v',w', x2,y2,z2,...
								output_velfluc_VX << x_ijk << " " << y_ijk << " " << z_ijk << " " << tmp_data[kap+0*vol] << " " << tmp_data[kap+1*vol] << " " << tmp_data[kap+2*vol] << " " << tmp_data[kap+3*vol] << " ";
							}
						}
					}
					output_velfluc_VX << std::endl;
				}
				
				
				// Update time-averages.
				for (int kap = 0; kap < vol; kap++)
				{
					// Reynolds stress.
					for (int d2 = 0; d2 < 3; d2++)
					{
						for (int d1 = 0; d1 < 3; d1++)
						{
							int p = d1 + 3*d2;
							tmp_data_rs[t][kap+p*vol] = ((double)K*tmp_data_rs[t][kap+p*vol] + tmp_data[kap+d1*vol]*tmp_data[kap+d2*vol]) / ( (double)(K+1) );
						}
					}
				}
				
				// Free allocations.
				delete[] tmp_data;
			}
			
			}
		}
		
		
		// Combine time-averaged datasets.
		for (int kap = 0; kap < 9*vol; kap++)
			tmp_data_rs[0][kap] = ((double)N_OUTPUT/(double)N_PROCS)*tmp_data_rs[0][kap];
		for (int K = 1; K < N_PROCS; K++)
		{
			// Final values stored in first array, then normalized afterward.
			for (int kap = 0; kap < 9*vol; kap++)
				tmp_data_rs[0][kap] += ((double)N_OUTPUT/(double)N_PROCS)*tmp_data_rs[K][kap];
		}
		for (int kap = 0; kap < 9*vol; kap++)
			tmp_data_rs[0][kap] /= (double)N_OUTPUT;
		
		
		// o====================================================================================
		// | Compute remaining properties (e.g., Turbulent kinetic energy, dissipation...).
		// o====================================================================================
		
		if (verbose)
			std::cout << "[-] Computing properties..." << std::endl;
		for (int k = 0; k < Nxi_f[2]; k++)
		{
			for (int j = 0; j < Nxi_f[1]; j++)
			{
				for (int i = 0; i < Nxi_f[0]; i++)
				{
					long int kap = i + Nxi_f[0]*j + Nxi_f[0]*Nxi_f[1]*k;
					
					// Mean velocity field.
					tmp_data_rs[0][kap+9*vol] = tmp_data_ave[0][kap+1*vol];
					tmp_data_rs[0][kap+10*vol] = tmp_data_ave[0][kap+2*vol];
					tmp_data_rs[0][kap+11*vol] = tmp_data_ave[0][kap+3*vol];
					
					// Mean pressure gradient vector.
					double pressure_grad[3];
					ScalGrad(i,j,k,13,   Nxi_f,dx_f,vol,tmp_data_ave[0],pressure_grad);
					tmp_data_rs[0][kap+12*vol] = pressure_grad[0];
					tmp_data_rs[0][kap+13*vol] = pressure_grad[1];
					tmp_data_rs[0][kap+14*vol] = pressure_grad[2];
					
					// Turbulent kinetic energy.
					tmp_data_rs[0][kap+15*vol] = 0.5*(tmp_data_rs[0][kap+0*vol] + tmp_data_rs[0][kap+4*vol] + tmp_data_rs[0][kap+8*vol]);
					
					// Turbulence kinetic energy production components.
					double vel_grad[9];
					Eigen::Matrix3f m_velgrad(3,3);
					VelGrad(i,j,k,1,   Nxi_f,dx_f,vol,tmp_data_ave[0],vel_grad,&m_velgrad);
					tmp_data_rs[0][kap+16*vol] = 0.0;
					for (int p = 0; p < 9; p++)
						tmp_data_rs[0][kap+16*vol] -= tmp_data_rs[0][kap+p*vol]*vel_grad[p];
					tmp_data_rs[0][kap+16*vol] *= 1.0/(0.05*0.05*0.05);
				}
			}
		}
		if (verbose)
		{
			std::cout << "    Finished computing properties..." << std::endl;
			std::cout << std::endl;
		}
		
		
		// o====================================================================================
		// | VTK array setup if either rendering or writing selected.
		// o====================================================================================
		
		
		{
			// Define and fill VTK arrays.
				// Reynolds Stresses.
			vtkNew<vtkDoubleArray> cell_data_RS;
			cell_data_RS->SetName("Reynolds Stresses");
			cell_data_RS->SetNumberOfComponents(9);
			cell_data_RS->SetNumberOfTuples(vol);
				// TA Velocity.
			vtkNew<vtkDoubleArray> cell_data_velocity;
			cell_data_velocity->SetName("TA Velocity");
			cell_data_velocity->SetNumberOfComponents(3);
			cell_data_velocity->SetNumberOfTuples(vol);
				// TA Pressure Gradient.
			vtkNew<vtkDoubleArray> cell_data_PG;
			cell_data_PG->SetName("TA Pressure Gradient");
			cell_data_PG->SetNumberOfComponents(3);
			cell_data_PG->SetNumberOfTuples(vol);
				// TKE.
			vtkNew<vtkDoubleArray> cell_data_TKE;
			cell_data_TKE->SetName("TKE");
			cell_data_TKE->SetNumberOfComponents(1);
			cell_data_TKE->SetNumberOfTuples(vol);
				// TKE Production.
			vtkNew<vtkDoubleArray> cell_data_TKEP;
			cell_data_TKEP->SetName("TKE Production");
			cell_data_TKEP->SetNumberOfComponents(1);
			cell_data_TKEP->SetNumberOfTuples(vol);
				// AMR Level.
			vtkNew<vtkDoubleArray> cell_data_level;
			cell_data_level->SetName("AMR Level");
			cell_data_level->SetNumberOfComponents(1);
			cell_data_level->SetNumberOfTuples(vol);
				// Block Id.
			vtkNew<vtkDoubleArray> cell_data_blockid;
			cell_data_blockid->SetName("Block Id");
			cell_data_blockid->SetNumberOfComponents(1);
			cell_data_blockid->SetNumberOfTuples(vol);
			// |
			if (verbose)
				std::cout << "[-] Inserting data in VTK pointers..." << std::endl;
			for (long int kap = 0; kap < vol; kap++)
			{
				cell_data_RS->SetTuple9(kap,
					tmp_data_rs[0][kap+ 0*vol],
					tmp_data_rs[0][kap+ 1*vol],
					tmp_data_rs[0][kap+ 2*vol],
					tmp_data_rs[0][kap+ 3*vol],
					tmp_data_rs[0][kap+ 4*vol],
					tmp_data_rs[0][kap+ 5*vol],
					tmp_data_rs[0][kap+ 6*vol],
					tmp_data_rs[0][kap+ 7*vol],
					tmp_data_rs[0][kap+ 8*vol]
				);
				cell_data_velocity->SetTuple3(kap,
					tmp_data_rs[0][kap+ 9*vol],
					tmp_data_rs[0][kap+ 10*vol],
					tmp_data_rs[0][kap+ 11*vol]
				);
				cell_data_PG->SetTuple3(kap, 
					tmp_data_rs[0][kap+ 12*vol],
					tmp_data_rs[0][kap+ 13*vol],
					tmp_data_rs[0][kap+ 14*vol]
				);
				cell_data_TKE->SetTuple1(kap, 
					tmp_data_rs[0][kap+ 15*vol]
				);
	// 			cell_data_TKEP->SetTuple9(kap, 
	// 				tmp_data_rs[0][kap+ 16*vol],
	// 				tmp_data_rs[0][kap+ 17*vol],
	// 			     	tmp_data_rs[0][kap+ 18*vol],
	// 			     	tmp_data_rs[0][kap+ 19*vol],
	// 			     	tmp_data_rs[0][kap+ 20*vol],
	// 			     	tmp_data_rs[0][kap+ 21*vol],
	// 			     	tmp_data_rs[0][kap+ 22*vol],
	// 			     	tmp_data_rs[0][kap+ 23*vol],
	// 			     	tmp_data_rs[0][kap+ 24*vol]
	// 			);
				cell_data_TKEP->SetTuple1(kap,
					tmp_data_rs[0][kap+ 16*vol]
				);
				cell_data_level->SetTuple1(kap,
					tmp_data_rs[0][kap+ 17*vol]
				);
				cell_data_blockid->SetTuple1(kap,
					tmp_data_rs[0][kap+ 18*vol]
				);
			}
			if (verbose)
				std::cout << "    Finished inserting data in VTK pointers..." << std::endl;
		
		
			// o====================================================================================
			// | Write to .vti file, if selected.
			// o====================================================================================
			
			
			if (verbose)
				std::cout << "[-] Creating uniform grid..." << std::endl;
				// Parameters and initialization.
			double origin[3] = {VOL_I_MIN*Nbx*dx0, VOL_J_MIN*Nbx*dx0, VOL_K_MIN*Nbx*dx0};
			double spacing[3] = {dx_f, dx_f, dx_f};
			vtkNew<vtkUniformGrid> grid;
				// Set up image data grid.
			grid->Initialize();
			grid->SetOrigin(origin);
			grid->SetSpacing(spacing);
			grid->SetDimensions(Nxi_f[0]+1, Nxi_f[1]+1, N_DIM==2?2:Nxi_f[2]+1);
			grid->GetCellData()->AddArray(cell_data_RS);
			grid->GetCellData()->AddArray(cell_data_velocity);
			grid->GetCellData()->AddArray(cell_data_PG);
			grid->GetCellData()->AddArray(cell_data_TKE);
			grid->GetCellData()->AddArray(cell_data_TKEP);
			grid->GetCellData()->AddArray(cell_data_level);
			grid->GetCellData()->AddArray(cell_data_blockid);
				// Blank invalid cells (these are identified by negative AMR level).
			grid->AllocateCellGhostArray();
			vtkUnsignedCharArray *ghosts = grid->GetCellGhostArray();
			for (long int kap = 0; kap < vol; kap++)
			{
				if (tmp_data_rs[0][kap + 17*vol] < 0)
					ghosts->SetValue(kap, ghosts->GetValue(kap) | vtkDataSetAttributes::HIDDENCELL);
			}
			if (verbose)
				std::cout << "    Finished creating uniform grid..." << std::endl;
			//|
			if (verbose)
				std::cout << "Finished building VTK dataset, writing..." << std::endl;
			std::string file_name = dirname + std::string("turb_out.vti");
			vtkNew<vtkXMLImageDataWriter> writer;
			writer->SetInputData(grid);
			writer->SetFileName(file_name.c_str());
			writer->Write();
			if (verbose)
				std::cout << "Finished writing VTK dataset..." << std::endl;
		}
		
		// Close direct-output file.
		output_velfluc_UZ.close();
		output_velfluc_VX.close();
	}




	// Free allocations.
	delete[] mult_f;
	delete[] buffer;
	for (int K = 0; K < N_PROCS; K++)
		delete[] tmp_data_ave[K];
	
	return 0;
}
