/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#include "geometry.h"

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Geometry<ufloat_t,ufloat_g_t,AP>::G_AddBoundingBox(ufloat_g_t ax, ufloat_g_t bx, ufloat_g_t ay, ufloat_g_t by, ufloat_g_t az, ufloat_g_t bz)
{
	int N_nodes_curr = v_geom_f_node_X.size();
	
	if (N_DIM==2)
	{
		// Bottom-left.
		v_geom_f_node_X.push_back(ax);
		v_geom_f_node_Y.push_back(ay);
		v_geom_f_node_Z.push_back(az);
		v_geom_ID_face_1.push_back( N_nodes_curr + 0 );
		v_geom_ID_face_2.push_back( N_nodes_curr + 1 );
		v_geom_ID_face_3.push_back( 0 );
		
		
		// Top-left.
		v_geom_f_node_X.push_back(ax);
		v_geom_f_node_Y.push_back(by);
		v_geom_f_node_Z.push_back(az);
		v_geom_ID_face_1.push_back( N_nodes_curr + 1 );
		v_geom_ID_face_2.push_back( N_nodes_curr + 2 );
		v_geom_ID_face_3.push_back( 0 );
		
		// Top-right.
		v_geom_f_node_X.push_back(bx);
		v_geom_f_node_Y.push_back(by);
		v_geom_f_node_Z.push_back(az);
		v_geom_ID_face_1.push_back( N_nodes_curr + 2 );
		v_geom_ID_face_2.push_back( N_nodes_curr + 3 );
		v_geom_ID_face_3.push_back( 0 );
		
		// Bottom-right.
		v_geom_f_node_X.push_back(bx);
		v_geom_f_node_Y.push_back(ay);
		v_geom_f_node_Z.push_back(az);
		v_geom_ID_face_1.push_back( N_nodes_curr + 3 );
		v_geom_ID_face_2.push_back( N_nodes_curr + 0 );
		v_geom_ID_face_3.push_back( 0 );
	}
	else
	{
		// Nodes.
		v_geom_f_node_X.push_back(ax); // 0
		v_geom_f_node_Y.push_back(ay);
		v_geom_f_node_Z.push_back(az);
		v_geom_f_node_X.push_back(ax); // 1
		v_geom_f_node_Y.push_back(by);
		v_geom_f_node_Z.push_back(az);
		v_geom_f_node_X.push_back(ax); // 2
		v_geom_f_node_Y.push_back(ay);
		v_geom_f_node_Z.push_back(bz);
		v_geom_f_node_X.push_back(ax); // 3
		v_geom_f_node_Y.push_back(by);
		v_geom_f_node_Z.push_back(bz);
		v_geom_f_node_X.push_back(bx); // 4
		v_geom_f_node_Y.push_back(ay);
		v_geom_f_node_Z.push_back(az);
		v_geom_f_node_X.push_back(bx); // 5
		v_geom_f_node_Y.push_back(by);
		v_geom_f_node_Z.push_back(az);
		v_geom_f_node_X.push_back(bx); // 6
		v_geom_f_node_Y.push_back(ay);
		v_geom_f_node_Z.push_back(bz);
		v_geom_f_node_X.push_back(bx); // 7
		v_geom_f_node_Y.push_back(by);
		v_geom_f_node_Z.push_back(bz);
		
		// -X
		v_geom_ID_face_1.push_back( N_nodes_curr + 0 );
		v_geom_ID_face_2.push_back( N_nodes_curr + 1 );
		v_geom_ID_face_3.push_back( N_nodes_curr + 3 );
		v_geom_ID_face_1.push_back( N_nodes_curr + 0 );
		v_geom_ID_face_2.push_back( N_nodes_curr + 3 );
		v_geom_ID_face_3.push_back( N_nodes_curr + 2 );
		
		// +X
		v_geom_ID_face_1.push_back( N_nodes_curr + 4 );
		v_geom_ID_face_2.push_back( N_nodes_curr + 6 );
		v_geom_ID_face_3.push_back( N_nodes_curr + 7 );
		v_geom_ID_face_1.push_back( N_nodes_curr + 4 );
		v_geom_ID_face_2.push_back( N_nodes_curr + 7 );
		v_geom_ID_face_3.push_back( N_nodes_curr + 5 );
		
		// -Y
		v_geom_ID_face_1.push_back( N_nodes_curr + 0 );
		v_geom_ID_face_2.push_back( N_nodes_curr + 2 );
		v_geom_ID_face_3.push_back( N_nodes_curr + 6 );
		v_geom_ID_face_1.push_back( N_nodes_curr + 0 );
		v_geom_ID_face_2.push_back( N_nodes_curr + 6 );
		v_geom_ID_face_3.push_back( N_nodes_curr + 4 );
		
		// +Y
		v_geom_ID_face_1.push_back( N_nodes_curr + 1 );
		v_geom_ID_face_2.push_back( N_nodes_curr + 5 );
		v_geom_ID_face_3.push_back( N_nodes_curr + 7 );
		v_geom_ID_face_1.push_back( N_nodes_curr + 1 );
		v_geom_ID_face_2.push_back( N_nodes_curr + 7 );
		v_geom_ID_face_3.push_back( N_nodes_curr + 3 );
		
		// -Z
		v_geom_ID_face_1.push_back( N_nodes_curr + 7 );
		v_geom_ID_face_2.push_back( N_nodes_curr + 6 );
		v_geom_ID_face_3.push_back( N_nodes_curr + 2 );
		v_geom_ID_face_1.push_back( N_nodes_curr + 7 );
		v_geom_ID_face_2.push_back( N_nodes_curr + 2 );
		v_geom_ID_face_3.push_back( N_nodes_curr + 3 );
		
		// +Z
		v_geom_ID_face_1.push_back( N_nodes_curr + 0 );
		v_geom_ID_face_2.push_back( N_nodes_curr + 4 );
		v_geom_ID_face_3.push_back( N_nodes_curr + 5 );
		v_geom_ID_face_1.push_back( N_nodes_curr + 0 );
		v_geom_ID_face_2.push_back( N_nodes_curr + 5 );
		v_geom_ID_face_3.push_back( N_nodes_curr + 1 );
	}
	
	return 0;
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Geometry<ufloat_t,ufloat_g_t,AP>::G_AddRectangle(ufloat_g_t ax, ufloat_g_t bx, ufloat_g_t ay, ufloat_g_t by)
{
	int N_nodes_curr = v_geom_f_node_X.size();
	
	// Bottom-left.
	v_geom_f_node_X.push_back(ax);
	v_geom_f_node_Y.push_back(ay);
	v_geom_f_node_Z.push_back(0.0);
	v_geom_ID_face_1.push_back( N_nodes_curr + 0 );
	v_geom_ID_face_2.push_back( N_nodes_curr + 1 );
	v_geom_ID_face_3.push_back( 0 );
	
	// Bottom-right.
	v_geom_f_node_X.push_back(bx);
	v_geom_f_node_Y.push_back(ay);
	v_geom_f_node_Z.push_back(0.0);
	v_geom_ID_face_1.push_back( N_nodes_curr + 1 );
	v_geom_ID_face_2.push_back( N_nodes_curr + 2 );
	v_geom_ID_face_3.push_back( 0 );
	
	// Top-right.
	v_geom_f_node_X.push_back(bx);
	v_geom_f_node_Y.push_back(by);
	v_geom_f_node_Z.push_back(0.0);
	v_geom_ID_face_1.push_back( N_nodes_curr + 2 );
	v_geom_ID_face_2.push_back( N_nodes_curr + 3 );
	v_geom_ID_face_3.push_back( 0 );
	
	// Top-left.
	v_geom_f_node_X.push_back(ax);
	v_geom_f_node_Y.push_back(by);
	v_geom_f_node_Z.push_back(0.0);
	v_geom_ID_face_1.push_back( N_nodes_curr + 3 );
	v_geom_ID_face_2.push_back( N_nodes_curr + 0 );
	v_geom_ID_face_3.push_back( 0 );
	
	return 0;
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Geometry<ufloat_t,ufloat_g_t,AP>::G_AddCircle(int N, ufloat_g_t cx, ufloat_g_t cy, ufloat_g_t R)
{
	ufloat_g_t pi = M_PI;
	ufloat_g_t denom = (ufloat_g_t)N-1.0;
	int N_nodes_curr = v_geom_f_node_X.size();
	
	for (int j = 0; j < N; j++)
	{
		ufloat_g_t t_j = (2*pi/denom)*j;
		ufloat_g_t x_j = R*cos(t_j) + cx;
		ufloat_g_t y_j = R*sin(t_j) + cy;
		ufloat_g_t z_j = 0.0;
		v_geom_f_node_X.push_back(x_j);
		v_geom_f_node_Y.push_back(y_j);
		v_geom_f_node_Z.push_back(z_j);
		
		if (j < N-1)
		{
			v_geom_ID_face_1.push_back( N_nodes_curr + j );
			v_geom_ID_face_2.push_back( N_nodes_curr + j+1 );
			v_geom_ID_face_3.push_back( 0 );
		}
	}
	
	return 0;
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Geometry<ufloat_t,ufloat_g_t,AP>::G_AddPrism(ufloat_g_t ax, ufloat_g_t bx, ufloat_g_t ay, ufloat_g_t by, ufloat_g_t az, ufloat_g_t bz)
{
	int N_nodes_curr = v_geom_f_node_X.size();
	
	// Nodes.
	v_geom_f_node_X.push_back(ax); // 0
	v_geom_f_node_Y.push_back(ay);
	v_geom_f_node_Z.push_back(az);
	v_geom_f_node_X.push_back(ax); // 1
	v_geom_f_node_Y.push_back(by);
	v_geom_f_node_Z.push_back(az);
	v_geom_f_node_X.push_back(ax); // 2
	v_geom_f_node_Y.push_back(ay);
	v_geom_f_node_Z.push_back(bz);
	v_geom_f_node_X.push_back(ax); // 3
	v_geom_f_node_Y.push_back(by);
	v_geom_f_node_Z.push_back(bz);
	v_geom_f_node_X.push_back(bx); // 4
	v_geom_f_node_Y.push_back(ay);
	v_geom_f_node_Z.push_back(az);
	v_geom_f_node_X.push_back(bx); // 5
	v_geom_f_node_Y.push_back(by);
	v_geom_f_node_Z.push_back(az);
	v_geom_f_node_X.push_back(bx); // 6
	v_geom_f_node_Y.push_back(ay);
	v_geom_f_node_Z.push_back(bz);
	v_geom_f_node_X.push_back(bx); // 7
	v_geom_f_node_Y.push_back(by);
	v_geom_f_node_Z.push_back(bz);
	
	// -X
	v_geom_ID_face_1.push_back( N_nodes_curr + 0 );
	v_geom_ID_face_2.push_back( N_nodes_curr + 2 );
	v_geom_ID_face_3.push_back( N_nodes_curr + 3 );
	v_geom_ID_face_1.push_back( N_nodes_curr + 0 );
	v_geom_ID_face_2.push_back( N_nodes_curr + 3 );
	v_geom_ID_face_3.push_back( N_nodes_curr + 1 );
	
	// +X
	v_geom_ID_face_1.push_back( N_nodes_curr + 4 );
	v_geom_ID_face_2.push_back( N_nodes_curr + 5 );
	v_geom_ID_face_3.push_back( N_nodes_curr + 7 );
	v_geom_ID_face_1.push_back( N_nodes_curr + 4 );
	v_geom_ID_face_2.push_back( N_nodes_curr + 7 );
	v_geom_ID_face_3.push_back( N_nodes_curr + 6 );
	
	// -Y
	v_geom_ID_face_1.push_back( N_nodes_curr + 0 );
	v_geom_ID_face_2.push_back( N_nodes_curr + 4 );
	v_geom_ID_face_3.push_back( N_nodes_curr + 6 );
	v_geom_ID_face_1.push_back( N_nodes_curr + 0 );
	v_geom_ID_face_2.push_back( N_nodes_curr + 6 );
	v_geom_ID_face_3.push_back( N_nodes_curr + 2 );
	
	// +Y
	v_geom_ID_face_1.push_back( N_nodes_curr + 1 );
	v_geom_ID_face_2.push_back( N_nodes_curr + 3 );
	v_geom_ID_face_3.push_back( N_nodes_curr + 7 );
	v_geom_ID_face_1.push_back( N_nodes_curr + 1 );
	v_geom_ID_face_2.push_back( N_nodes_curr + 7 );
	v_geom_ID_face_3.push_back( N_nodes_curr + 5 );
		
	// -Z
	v_geom_ID_face_1.push_back( N_nodes_curr + 7 );
	v_geom_ID_face_2.push_back( N_nodes_curr + 3 );
	v_geom_ID_face_3.push_back( N_nodes_curr + 2 );
	v_geom_ID_face_1.push_back( N_nodes_curr + 7 );
	v_geom_ID_face_2.push_back( N_nodes_curr + 2 );
	v_geom_ID_face_3.push_back( N_nodes_curr + 6 );
	
	// +Z
	v_geom_ID_face_1.push_back( N_nodes_curr + 0 );
	v_geom_ID_face_2.push_back( N_nodes_curr + 1 );
	v_geom_ID_face_3.push_back( N_nodes_curr + 5 );
	v_geom_ID_face_1.push_back( N_nodes_curr + 0 );
	v_geom_ID_face_2.push_back( N_nodes_curr + 5 );
	v_geom_ID_face_3.push_back( N_nodes_curr + 4 );
	
	return 0;
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Geometry<ufloat_t,ufloat_g_t,AP>::G_AddSphere(int N1, int N2, ufloat_g_t cx, ufloat_g_t cy, ufloat_g_t cz, ufloat_g_t R)
{
	ufloat_g_t pi = M_PI;
	ufloat_g_t denom1 = (ufloat_g_t)N1-1.0;
	ufloat_g_t denom2 = (ufloat_g_t)N2-1.0;
	int N_nodes_curr = v_geom_f_node_X.size();
	
	// N1 and N2 are resolutions in the polar (0-pi) and azimuthal (0-2*pi) axes, respectively.
	for (int j = 0; j < N1; j++)
	{
		for (int k = 0; k < N2; k++)
		{
			ufloat_g_t t_jk = (pi/denom1)*j;
			ufloat_g_t p_jk = (2*pi/denom2)*k;
			ufloat_g_t x_jk = R*sin(t_jk)*cos(p_jk) + cx;
			ufloat_g_t y_jk = R*sin(t_jk)*sin(p_jk) + cy;
			ufloat_g_t z_jk = R*cos(t_jk) + cz;
			v_geom_f_node_X.push_back(x_jk);
			v_geom_f_node_Y.push_back(y_jk);
			v_geom_f_node_Z.push_back(z_jk);
			
			if (j < N1-1 && k < N2-1)
			{
				v_geom_ID_face_1.push_back( N_nodes_curr + ((k+0)+N2*(j+0)) );
				v_geom_ID_face_3.push_back( N_nodes_curr + ((k+1)+N2*(j+0)) );
				v_geom_ID_face_2.push_back( N_nodes_curr + ((k+1)+N2*(j+1)) );
				
				v_geom_ID_face_1.push_back( N_nodes_curr + ((k+0)+N2*(j+0)) );
				v_geom_ID_face_3.push_back( N_nodes_curr + ((k+1)+N2*(j+1)) );
				v_geom_ID_face_2.push_back( N_nodes_curr + ((k+0)+N2*(j+1)) );
			}
		}
	}
	
	return 0;
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Geometry<ufloat_t,ufloat_g_t,AP>::G_AddNACA002D(int N, ufloat_g_t t, ufloat_g_t ax, ufloat_g_t bx, ufloat_g_t ay, ufloat_g_t by, int te)
{
	ufloat_g_t denom = (ufloat_g_t)N-1.0;
	ufloat_g_t last_coeff = -0.1036;
	if (te)
		last_coeff = -0.1015;
	int N_nodes_curr = v_geom_f_node_X.size();
	
	// Nodes.
	for (int j = 0; j < N; j++)
	{
		ufloat_g_t x_j = ax + ((bx-ax)/denom)*j;
		ufloat_g_t x_jp = (1.0/denom)*j;
		ufloat_g_t y_j = (ay+by)/2.0 + 5*t*(0.2969*sqrt(x_jp) + x_jp*(-0.1260 + x_jp*(-0.3516 + x_jp*(0.2843 + x_jp*(last_coeff)))));
		ufloat_g_t z_j = 0.0;
		v_geom_f_node_X.push_back(x_j);
		v_geom_f_node_Y.push_back(y_j);
		v_geom_f_node_Z.push_back(z_j);
	}
	for (int j = 0; j < N; j++)
	{
		ufloat_g_t x_j = ax + ((bx-ax)/denom)*j;
		ufloat_g_t x_jp = (1.0/denom)*j;
		ufloat_g_t y_j = (ay+by)/2.0 + -5*t*(0.2969*sqrt(x_jp) + x_jp*(-0.1260 + x_jp*(-0.3516 + x_jp*(0.2843 + x_jp*(last_coeff)))));
		ufloat_g_t z_j = 0.0;
		v_geom_f_node_X.push_back(x_j);
		v_geom_f_node_Y.push_back(y_j);
		v_geom_f_node_Z.push_back(z_j);
	}
	
	// Upper edge.
	for (int j = N-1; j > 0; j--)
	{
		v_geom_ID_face_1.push_back( N_nodes_curr + 0+(j+0) );
		v_geom_ID_face_2.push_back( N_nodes_curr + 0+(j-1) );
		v_geom_ID_face_3.push_back( 0 );
	}
	
	// Lower edge.
	for (int j = 0; j < N-1; j++)
	{
		v_geom_ID_face_1.push_back( N_nodes_curr + N+(j+0) );
		v_geom_ID_face_2.push_back( N_nodes_curr + N+(j+1) );
		v_geom_ID_face_3.push_back( 0 );
	}
	
	// Trailing edge, if applicable.
	if (te)
	{
		v_geom_ID_face_1.push_back( N_nodes_curr + 2*N-1 );
		v_geom_ID_face_2.push_back( N_nodes_curr + N-1 );
		v_geom_ID_face_3.push_back( 0 );
	}
	
	return 0;
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Geometry<ufloat_t,ufloat_g_t,AP>::G_AddNACA003D(int N, ufloat_g_t t, ufloat_g_t ax, ufloat_g_t bx, ufloat_g_t ay, ufloat_g_t by, ufloat_g_t az, ufloat_g_t bz, int te)
{
	ufloat_g_t denom = (ufloat_g_t)N-1.0;
	ufloat_g_t last_coeff = -0.1036;
	if (te)
		last_coeff = -0.1015;
	int N_nodes_curr = v_geom_f_node_X.size();
	
	// Nodes.
	for (int j = 0; j < N; j++)
	{
		ufloat_g_t x_j = ax + ((bx-ax)/denom)*j;
		ufloat_g_t y_j = by;
		ufloat_g_t z_j = (bz+az)/2.0;
		v_geom_f_node_X.push_back(x_j);
		v_geom_f_node_Y.push_back(y_j);
		v_geom_f_node_Z.push_back(z_j);
	}
	for (int j = 0; j < N; j++)
	{
		ufloat_g_t x_j = ax + ((bx-ax)/denom)*j;
		ufloat_g_t x_jp = (1.0/denom)*j;
		ufloat_g_t y_j = by;
		//ufloat_g_t z_j = bz;
		ufloat_g_t z_j = (bz+az)/2.0 + 5*t*(0.2969*sqrt(x_jp) + x_jp*(-0.1260 + x_jp*(-0.3516 + x_jp*(0.2843 + x_jp*(last_coeff)))));
		v_geom_f_node_X.push_back(x_j);
		v_geom_f_node_Y.push_back(y_j);
		v_geom_f_node_Z.push_back(z_j);
	}
	for (int j = 0; j < N; j++)
	{
		ufloat_g_t x_j = ax + ((bx-ax)/denom)*j;
		ufloat_g_t x_jp = (1.0/denom)*j;
		ufloat_g_t y_j = by;
		//ufloat_g_t z_j = az;
		ufloat_g_t z_j = (bz+az)/2.0 + -5*t*(0.2969*sqrt(x_jp) + x_jp*(-0.1260 + x_jp*(-0.3516 + x_jp*(0.2843 + x_jp*(last_coeff)))));
		v_geom_f_node_X.push_back(x_j);
		v_geom_f_node_Y.push_back(y_j);
		v_geom_f_node_Z.push_back(z_j);
	}
	for (int j = 0; j < N; j++)
	{
		ufloat_g_t x_j = ax + ((bx-ax)/denom)*j;
		ufloat_g_t y_j = ay;
		ufloat_g_t z_j = (bz+az)/2.0;
		v_geom_f_node_X.push_back(x_j);
		v_geom_f_node_Y.push_back(y_j);
		v_geom_f_node_Z.push_back(z_j);
	}
	for (int j = 0; j < N; j++)
	{
		ufloat_g_t x_j = ax + ((bx-ax)/denom)*j;
		ufloat_g_t x_jp = (1.0/denom)*j;
		ufloat_g_t y_j = ay;
		//ufloat_g_t z_j = bz;
		ufloat_g_t z_j = (bz+az)/2.0 + 5*t*(0.2969*sqrt(x_jp) + x_jp*(-0.1260 + x_jp*(-0.3516 + x_jp*(0.2843 + x_jp*(last_coeff)))));
		v_geom_f_node_X.push_back(x_j);
		v_geom_f_node_Y.push_back(y_j);
		v_geom_f_node_Z.push_back(z_j);
	}
	for (int j = 0; j < N; j++)
	{
		ufloat_g_t x_j = ax + ((bx-ax)/denom)*j;
		ufloat_g_t x_jp = (1.0/denom)*j;
		ufloat_g_t y_j = ay;
		//ufloat_g_t z_j = az;
		ufloat_g_t z_j = (bz+az)/2.0 + -5*t*(0.2969*sqrt(x_jp) + x_jp*(-0.1260 + x_jp*(-0.3516 + x_jp*(0.2843 + x_jp*(last_coeff)))));
		v_geom_f_node_X.push_back(x_j);
		v_geom_f_node_Y.push_back(y_j);
		v_geom_f_node_Z.push_back(z_j);
	}
	
	// Right side.
	for (int j = 0; j < N-1; j++)
	{
		// Upper.
		v_geom_ID_face_1.push_back( N_nodes_curr + ((j+0)+N*(0)) );
		v_geom_ID_face_3.push_back( N_nodes_curr + ((j+1)+N*(0)) );
		v_geom_ID_face_2.push_back( N_nodes_curr + ((j+1)+N*(1)) );
		v_geom_ID_face_1.push_back( N_nodes_curr + ((j+0)+N*(0)) );
		v_geom_ID_face_3.push_back( N_nodes_curr + ((j+1)+N*(1)) );
		v_geom_ID_face_2.push_back( N_nodes_curr + ((j+0)+N*(1)) );
		
		// Lower.
		v_geom_ID_face_1.push_back( N_nodes_curr + ((j+0)+N*(0)) );
		v_geom_ID_face_3.push_back( N_nodes_curr + ((j+1)+N*(2)) );
		v_geom_ID_face_2.push_back( N_nodes_curr + ((j+1)+N*(0)) );
		v_geom_ID_face_1.push_back( N_nodes_curr + ((j+0)+N*(0)) );
		v_geom_ID_face_3.push_back( N_nodes_curr + ((j+0)+N*(2)) );
		v_geom_ID_face_2.push_back( N_nodes_curr + ((j+1)+N*(2)) );
	}
	
	// Left side.
	for (int j = 0; j < N-1; j++)
	{
		// Upper.
		v_geom_ID_face_1.push_back( N_nodes_curr + ((j+0)+N*(3)) );
		v_geom_ID_face_3.push_back( N_nodes_curr + ((j+1)+N*(4)) );
		v_geom_ID_face_2.push_back( N_nodes_curr + ((j+1)+N*(3)) );
		v_geom_ID_face_1.push_back( N_nodes_curr + ((j+0)+N*(3)) );
		v_geom_ID_face_3.push_back( N_nodes_curr + ((j+0)+N*(4)) );
		v_geom_ID_face_2.push_back( N_nodes_curr + ((j+1)+N*(4)) );
		
		// Lower.
		v_geom_ID_face_1.push_back( N_nodes_curr + ((j+0)+N*(3)) );
		v_geom_ID_face_3.push_back( N_nodes_curr + ((j+1)+N*(3)) );
		v_geom_ID_face_2.push_back( N_nodes_curr + ((j+1)+N*(5)) );
		v_geom_ID_face_1.push_back( N_nodes_curr + ((j+0)+N*(3)) );
		v_geom_ID_face_3.push_back( N_nodes_curr + ((j+1)+N*(5)) );
		v_geom_ID_face_2.push_back( N_nodes_curr + ((j+0)+N*(5)) );
	}
	
	// Top and bottom sides.
	for (int j = 0; j < N-1; j++)
	{
		// Upper.
		v_geom_ID_face_1.push_back( N_nodes_curr + ((j+0)+N*(1)) );
		v_geom_ID_face_3.push_back( N_nodes_curr + ((j+1)+N*(1)) );
		v_geom_ID_face_2.push_back( N_nodes_curr + ((j+1)+N*(4)) );
		v_geom_ID_face_1.push_back( N_nodes_curr + ((j+0)+N*(1)) );
		v_geom_ID_face_3.push_back( N_nodes_curr + ((j+1)+N*(4)) );
		v_geom_ID_face_2.push_back( N_nodes_curr + ((j+0)+N*(4)) );
		
		// Lower.
		v_geom_ID_face_1.push_back( N_nodes_curr + ((j+0)+N*(5)) );
		v_geom_ID_face_3.push_back( N_nodes_curr + ((j+1)+N*(5)) );
		v_geom_ID_face_2.push_back( N_nodes_curr + ((j+1)+N*(2)) );
		v_geom_ID_face_1.push_back( N_nodes_curr + ((j+0)+N*(5)) );
		v_geom_ID_face_3.push_back( N_nodes_curr + ((j+1)+N*(2)) );
		v_geom_ID_face_2.push_back( N_nodes_curr + ((j+0)+N*(2)) );
	}
	
	// Trailing edge, if applicable.
	if (te)
	{
		v_geom_ID_face_1.push_back( N_nodes_curr + ((N-1)+N*(0)) );
		v_geom_ID_face_3.push_back( N_nodes_curr + ((N-1)+N*(3)) );
		v_geom_ID_face_2.push_back( N_nodes_curr + ((N-1)+N*(4)) );
		v_geom_ID_face_1.push_back( N_nodes_curr + ((N-1)+N*(0)) );
		v_geom_ID_face_3.push_back( N_nodes_curr + ((N-1)+N*(4)) );
		v_geom_ID_face_2.push_back( N_nodes_curr + ((N-1)+N*(1)) );
		v_geom_ID_face_1.push_back( N_nodes_curr + ((N-1)+N*(2)) );
		v_geom_ID_face_3.push_back( N_nodes_curr + ((N-1)+N*(5)) );
		v_geom_ID_face_2.push_back( N_nodes_curr + ((N-1)+N*(3)) );
		v_geom_ID_face_1.push_back( N_nodes_curr + ((N-1)+N*(2)) );
		v_geom_ID_face_3.push_back( N_nodes_curr + ((N-1)+N*(3)) );
		v_geom_ID_face_2.push_back( N_nodes_curr + ((N-1)+N*(0)) );
	}
	
	return 0;
}

/*
int PrintSTL
(
	std::string output_dir,
	std::vector<ufloat_g_t> &v_geom_f_node_X,
	std::vector<ufloat_g_t> &v_geom_f_node_Y,
	std::vector<ufloat_g_t> &v_geom_f_node_Z,
	std::vector<int> &v_geom_ID_face_1,
	std::vector<int> &v_geom_ID_face_2,
	std::vector<int> &v_geom_ID_face_3
)
{
	std::ofstream stl = std::ofstream(output_dir + "geometry.stl");
	stl << "solid Geometry" << std::endl;
	
	// Compute facet normals and print.
	for (int i = 0; i < v_geom_ID_face_1.size(); i++)
	{
		int p1 = v_geom_ID_face_1[i];
		int p2 = v_geom_ID_face_2[i];
		int p3 = v_geom_ID_face_3[i];
		ufloat_g_t v1x = v_geom_f_node_X[p1];
		ufloat_g_t v1y = v_geom_f_node_Y[p1];
		ufloat_g_t v1z = v_geom_f_node_Z[p1];
		ufloat_g_t v2x = v_geom_f_node_X[p2];
		ufloat_g_t v2y = v_geom_f_node_Y[p2];
		ufloat_g_t v2z = v_geom_f_node_Z[p2];
		ufloat_g_t v3x = v_geom_f_node_X[p3];
		ufloat_g_t v3y = v_geom_f_node_Y[p3];
		ufloat_g_t v3z = v_geom_f_node_Z[p3];
#if (N_DIM==2)
		ufloat_g_t n1 = v2y-v1y;
		ufloat_g_t n2 = -(v2x-v1x);
		ufloat_g_t n3 = 0.0;
#else
		ufloat_g_t dx1 = v2x-v1x;
		ufloat_g_t dy1 = v2y-v1y;
		ufloat_g_t dz1 = v2z-v1z;
		ufloat_g_t dx2 = v3x-v1x;
		ufloat_g_t dy2 = v3y-v1y;
		ufloat_g_t dz2 = v3z-v1z;
		ufloat_g_t n1 = dy1*dz2-dz1*dy2;
		ufloat_g_t n2 = dz1*dx2-dx1*dz2;
		ufloat_g_t n3 = dx1*dy2-dy1*dx2;
#endif
		
		stl << "facet normal " << n1 << " " << n2 << " " << n3 << std::endl;
		stl << "    outer loop" << std::endl;
		stl << "        vertex " << v1x << " " << v1y << " " << v1z << std::endl;
		stl << "        vertex " << v2x << " " << v2y << " " << v2z << std::endl;
		stl << "        vertex " << v3x << " " << v3y << " " << v3z << std::endl;
		stl << "    endloop" << std::endl;
		stl << "endfacet" << std::endl;
	}
	
	stl << "endsolid Geometry" << std::endl;
	stl.close();
	
	return 0;
}

int PrintOBJ
(
	std::string output_dir,
	std::vector<ufloat_g_t> &v_geom_f_node_X,
	std::vector<ufloat_g_t> &v_geom_f_node_Y,
	std::vector<ufloat_g_t> &v_geom_f_node_Z,
	std::vector<int> &v_geom_ID_face_1,
	std::vector<int> &v_geom_ID_face_2,
	std::vector<int> &v_geom_ID_face_3
)
{
	std::ofstream obj = std::ofstream(output_dir + "geometry.obj");
	obj << "# Geometry." << std::endl << std::endl;;
	
	// Print vertices.
	obj << "Vertices." << std::endl;
	for (int i = 0; i < v_geom_f_node_X.size(); i++)
		obj << "v " << v_geom_f_node_X[i] << " " << v_geom_f_node_Y[i] << " " << v_geom_f_node_Z[i] << std::endl;
	obj << std::endl;
	
	// Print facs.
	obj << "Faces." << std::endl;
	for (int i = 0; i < v_geom_ID_face_1.size(); i++)
		obj << "f " << v_geom_ID_face_1[i]+1 << " " << v_geom_ID_face_2[i]+1 << " " << v_geom_ID_face_3[i]+1 << std::endl;
	
	obj.close();

	return 0;
}
*/

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Geometry<ufloat_t,ufloat_g_t,AP>::G_ClearVectors()
{
	v_geom_f_node_X.clear();
	v_geom_f_node_Y.clear();
	v_geom_f_node_Z.clear();
	v_geom_ID_face_1.clear();
	v_geom_ID_face_2.clear();
	v_geom_ID_face_3.clear();
	
	return 0;
}

template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Geometry<ufloat_t,ufloat_g_t,AP>::G_ImportBoundariesFromTextFile(int i_dev)
{
	// Add shapes defined in input file as necessary.
	std::cout << "[-] Reading geometry from input..." << std::endl;
	std::string word = "";
	std::ifstream input_geometry = std::ifstream(input_dir + "geometry.txt");
	for (std::string line; std::getline(input_geometry,line);)
	{
		if (line == "END")
			input_geometry.seekg(0,std::ios::end);
		else
		{
			std::stringstream ss(line);
			if (ss >> word)
			{
				if (word == "BOX")
				{
					G_AddBoundingBox(0, Lx, 0, Ly, 0, Lz);
					std::cout << "[-] Detected BOX, adding..." << std::endl;
				}
				if (word == "RECTANGLE")
				{
					std::string s;
					ufloat_g_t ax, bx, ay, by;
					ss >> s;
					if (s == "CORNER")
						ss >> ax >> bx >> ay >> by;
					else
					{
						ufloat_g_t cx, cy, lx, ly;
						ss >> cx >> cy >> lx >> ly;
						ax = cx-lx/2.0;
						bx = cx+lx/2.0;
						ay = cy-ly/2.0;
						by = cy+ly/2.0;
					}
					
					G_AddRectangle(ax, bx, ay, by);
					std::cout << "[-] Detected RECTANGLE, adding..." << std::endl;
				}
				if (word == "CIRCLE")
				{
					int N;
					ufloat_g_t cx, cy, R;
					ss >> N >> cx >> cy >> R;
					G_AddCircle(N, cx, cy, R);
					std::cout << "[-] Detected CIRCLE, adding..." << std::endl;
				}
				if (word == "PRISM")
				{
					std::string s;
					ufloat_g_t ax, bx, ay, by, az, bz;
					ss >> s;
					if (s == "CORNER")
						ss >> ax >> bx >> ay >> by >> az >> bz;
					else
					{
						ufloat_g_t cx, cy, cz, lx, ly, lz;
						ss >> cx >> cy >> cz >> lx >> ly >> lz;
						ax = cx-lx/2.0;
						bx = cx+lx/2.0;
						ay = cy-ly/2.0;
						by = cy+ly/2.0;
						az = cz-lz/2.0;
						bz = cz+lz/2.0;
					}
					
					G_AddPrism(ax, bx, ay, by, az, bz);
					std::cout << "[-] Detected PRISM, adding..." << std::endl;
				}
				if (word == "SPHERE")
				{
					int N1, N2;
					ufloat_g_t cx, cy, cz, R;
					ss >> N1 >> N2 >> cx >> cy >> cz >> R;
					G_AddSphere(N1, N2, cx, cy, cz, R);
					std::cout << "[-] Detected SPHERE, adding..." << std::endl;
				}
				if (word == "NACA002D")
				{
					std::string s;
					int N, te;
					ufloat_g_t t, ax, bx, ay, by;
					ss >> s;
					if (s == "CORNER")
						ss >> N >> t >> ax >> bx >> ay >> by >> te;
					else
					{
						ufloat_g_t cx, cy, lx, ly;
						ss >> N >> t >> cx >> cy >> lx >> ly >> te;
						ax = cx-lx/2.0;
						bx = cx+lx/2.0;
						ay = cy-ly/2.0;
						by = cy+ly/2.0;
					}
					
					G_AddNACA002D(N, t*0.01*(bx-ax), ax, bx, ay, by, te);
					std::cout << "[-] Detected NACA002D, adding..." << std::endl;
				}
				if (word == "NACA003D")
				{
					std::string s;
					int N, te;
					ufloat_g_t t, ax, bx, ay, by, az, bz;
					ss >> s;
					if (s == "CORNER")
						ss >> N >> t >> ax >> bx >> ay >> by >> az >> bz >> te;
					else
					{
						ufloat_g_t cx, cy, cz, lx, ly, lz;
						ss >> N >> t >> cx >> cy >> cz >> lx >> ly >> lz >> te;
						ax = cx-lx/2.0;
						bx = cx+lx/2.0;
						ay = cy-ly/2.0;
						by = cy+ly/2.0;
						az = cz-lz/2.0;
						bz = cz+lz/2.0;
					}
					
					G_AddNACA003D(N, t*0.01*(bx-ax), ax, bx, ay, by, az, bz, te);
					std::cout << "[-] Detected NACA003D, adding..." << std::endl;
				}
			}
		}
	}
	
	// Make node and face indices permanent. 
// 	n_nodes[i_dev] = v_geom_f_node_X.size();
// 	n_faces[i_dev] = v_geom_ID_face_1.size();
// 	n_nodes_a[i_dev] = n_nodes[i_dev] + 256-(n_nodes[i_dev]%256);
// 	n_faces_a[i_dev] = n_faces[i_dev] + 256-(n_faces[i_dev]%256);
	//G_Init_Arrays_CPU(i_dev);
	
	// Print the geometry to an STL file to superimpose over the computational grid in Paraview.
	//std::cout << "[-] Finished reading, print STL file..." << std::endl;
	//G_PrintOBJ(i_dev);
	//std::cout << "[-] Finished printing STL file..." << std::endl;
	
	return 0;
}

// int Geometry<float,float,AP>::G_AddBoundingBox(float ax, float bx, float ay, float by, float az, float bz);
// int Geometry<float,double,AP>::G_AddBoundingBox(double ax, double bx, double ay, double by, double az, double bz);
// int Geometry<double,float,AP>::G_AddBoundingBox(float ax, float bx, float ay, float by, float az, float bz);
// int Geometry<double,double,AP>::G_AddBoundingBox(double ax, double bx, double ay, double by, double az, double bz);
