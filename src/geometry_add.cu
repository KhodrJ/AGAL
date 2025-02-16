/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#include "mesh.h"

int AddBoundingBox
(
	std::vector<double> *v_geom_f_node_X,
	std::vector<double> *v_geom_f_node_Y,
	std::vector<double> *v_geom_f_node_Z,
	std::vector<int> *v_geom_ID_face_1,
	std::vector<int> *v_geom_ID_face_2,
	std::vector<int> *v_geom_ID_face_3,
	double ax, double bx, double ay, double by, double az, double bz
)
{
	int N_nodes_curr = (*v_geom_f_node_X).size();
	
	if (N_DIM==2)
	{
		// Bottom-left.
		(*v_geom_f_node_X).push_back(ax);
		(*v_geom_f_node_Y).push_back(ay);
		(*v_geom_f_node_Z).push_back(az);
		(*v_geom_ID_face_1).push_back( N_nodes_curr + 0 );
		(*v_geom_ID_face_2).push_back( N_nodes_curr + 1 );
		(*v_geom_ID_face_3).push_back( 0 );
		
		// Top-left.
		(*v_geom_f_node_X).push_back(ax);
		(*v_geom_f_node_Y).push_back(by);
		(*v_geom_f_node_Z).push_back(az);
		(*v_geom_ID_face_1).push_back( N_nodes_curr + 1 );
		(*v_geom_ID_face_2).push_back( N_nodes_curr + 2 );
		(*v_geom_ID_face_3).push_back( 0 );
		
		// Top-right.
		(*v_geom_f_node_X).push_back(bx);
		(*v_geom_f_node_Y).push_back(by);
		(*v_geom_f_node_Z).push_back(az);
		(*v_geom_ID_face_1).push_back( N_nodes_curr + 2 );
		(*v_geom_ID_face_2).push_back( N_nodes_curr + 3 );
		(*v_geom_ID_face_3).push_back( 0 );
		
		// Bottom-right.
		(*v_geom_f_node_X).push_back(bx);
		(*v_geom_f_node_Y).push_back(ay);
		(*v_geom_f_node_Z).push_back(az);
		(*v_geom_ID_face_1).push_back( N_nodes_curr + 3 );
		(*v_geom_ID_face_2).push_back( N_nodes_curr + 0 );
		(*v_geom_ID_face_3).push_back( 0 );
	}
	else
	{
		// Nodes.
		(*v_geom_f_node_X).push_back(ax); // 0
		(*v_geom_f_node_Y).push_back(ay);
		(*v_geom_f_node_Z).push_back(az);
		(*v_geom_f_node_X).push_back(ax); // 1
		(*v_geom_f_node_Y).push_back(by);
		(*v_geom_f_node_Z).push_back(az);
		(*v_geom_f_node_X).push_back(ax); // 2
		(*v_geom_f_node_Y).push_back(ay);
		(*v_geom_f_node_Z).push_back(bz);
		(*v_geom_f_node_X).push_back(ax); // 3
		(*v_geom_f_node_Y).push_back(by);
		(*v_geom_f_node_Z).push_back(bz);
		(*v_geom_f_node_X).push_back(bx); // 4
		(*v_geom_f_node_Y).push_back(ay);
		(*v_geom_f_node_Z).push_back(az);
		(*v_geom_f_node_X).push_back(bx); // 5
		(*v_geom_f_node_Y).push_back(by);
		(*v_geom_f_node_Z).push_back(az);
		(*v_geom_f_node_X).push_back(bx); // 6
		(*v_geom_f_node_Y).push_back(ay);
		(*v_geom_f_node_Z).push_back(bz);
		(*v_geom_f_node_X).push_back(bx); // 7
		(*v_geom_f_node_Y).push_back(by);
		(*v_geom_f_node_Z).push_back(bz);
		
		// -X
		(*v_geom_ID_face_1).push_back( N_nodes_curr + 0 );
		(*v_geom_ID_face_2).push_back( N_nodes_curr + 1 );
		(*v_geom_ID_face_3).push_back( N_nodes_curr + 3 );
		(*v_geom_ID_face_1).push_back( N_nodes_curr + 0 );
		(*v_geom_ID_face_2).push_back( N_nodes_curr + 3 );
		(*v_geom_ID_face_3).push_back( N_nodes_curr + 2 );
		
		// +X
		(*v_geom_ID_face_1).push_back( N_nodes_curr + 4 );
		(*v_geom_ID_face_2).push_back( N_nodes_curr + 6 );
		(*v_geom_ID_face_3).push_back( N_nodes_curr + 7 );
		(*v_geom_ID_face_1).push_back( N_nodes_curr + 4 );
		(*v_geom_ID_face_2).push_back( N_nodes_curr + 7 );
		(*v_geom_ID_face_3).push_back( N_nodes_curr + 5 );
		
		// -Y
		(*v_geom_ID_face_1).push_back( N_nodes_curr + 0 );
		(*v_geom_ID_face_2).push_back( N_nodes_curr + 2 );
		(*v_geom_ID_face_3).push_back( N_nodes_curr + 6 );
		(*v_geom_ID_face_1).push_back( N_nodes_curr + 0 );
		(*v_geom_ID_face_2).push_back( N_nodes_curr + 6 );
		(*v_geom_ID_face_3).push_back( N_nodes_curr + 4 );
		
		// +Y
		(*v_geom_ID_face_1).push_back( N_nodes_curr + 1 );
		(*v_geom_ID_face_2).push_back( N_nodes_curr + 5 );
		(*v_geom_ID_face_3).push_back( N_nodes_curr + 7 );
		(*v_geom_ID_face_1).push_back( N_nodes_curr + 1 );
		(*v_geom_ID_face_2).push_back( N_nodes_curr + 7 );
		(*v_geom_ID_face_3).push_back( N_nodes_curr + 3 );
		
		// -Z
		(*v_geom_ID_face_1).push_back( N_nodes_curr + 7 );
		(*v_geom_ID_face_2).push_back( N_nodes_curr + 6 );
		(*v_geom_ID_face_3).push_back( N_nodes_curr + 2 );
		(*v_geom_ID_face_1).push_back( N_nodes_curr + 7 );
		(*v_geom_ID_face_2).push_back( N_nodes_curr + 2 );
		(*v_geom_ID_face_3).push_back( N_nodes_curr + 3 );
		
		// +Z
		(*v_geom_ID_face_1).push_back( N_nodes_curr + 0 );
		(*v_geom_ID_face_2).push_back( N_nodes_curr + 4 );
		(*v_geom_ID_face_3).push_back( N_nodes_curr + 5 );
		(*v_geom_ID_face_1).push_back( N_nodes_curr + 0 );
		(*v_geom_ID_face_2).push_back( N_nodes_curr + 5 );
		(*v_geom_ID_face_3).push_back( N_nodes_curr + 1 );
	}
	
	return 0;
}

int AddRectangle
(
	std::vector<double> *v_geom_f_node_X,
	std::vector<double> *v_geom_f_node_Y,
	std::vector<double> *v_geom_f_node_Z,
	std::vector<int> *v_geom_ID_face_1,
	std::vector<int> *v_geom_ID_face_2,
	std::vector<int> *v_geom_ID_face_3,
	double ax, double bx, double ay, double by
)
{
	int N_nodes_curr = (*v_geom_f_node_X).size();
	
	// Bottom-left.
	(*v_geom_f_node_X).push_back(ax);
	(*v_geom_f_node_Y).push_back(ay);
	(*v_geom_f_node_Z).push_back(0.0);
	(*v_geom_ID_face_1).push_back( N_nodes_curr + 0 );
	(*v_geom_ID_face_2).push_back( N_nodes_curr + 1 );
	(*v_geom_ID_face_3).push_back( 0 );
	
	// Bottom-right.
	(*v_geom_f_node_X).push_back(bx);
	(*v_geom_f_node_Y).push_back(ay);
	(*v_geom_f_node_Z).push_back(0.0);
	(*v_geom_ID_face_1).push_back( N_nodes_curr + 1 );
	(*v_geom_ID_face_2).push_back( N_nodes_curr + 2 );
	(*v_geom_ID_face_3).push_back( 0 );
	
	// Top-right.
	(*v_geom_f_node_X).push_back(bx);
	(*v_geom_f_node_Y).push_back(by);
	(*v_geom_f_node_Z).push_back(0.0);
	(*v_geom_ID_face_1).push_back( N_nodes_curr + 2 );
	(*v_geom_ID_face_2).push_back( N_nodes_curr + 3 );
	(*v_geom_ID_face_3).push_back( 0 );
	
	// Top-left.
	(*v_geom_f_node_X).push_back(ax);
	(*v_geom_f_node_Y).push_back(by);
	(*v_geom_f_node_Z).push_back(0.0);
	(*v_geom_ID_face_1).push_back( N_nodes_curr + 3 );
	(*v_geom_ID_face_2).push_back( N_nodes_curr + 0 );
	(*v_geom_ID_face_3).push_back( 0 );
	
	return 0;
}

int AddCircle
(
	std::vector<double> *v_geom_f_node_X,
	std::vector<double> *v_geom_f_node_Y,
	std::vector<double> *v_geom_f_node_Z,
	std::vector<int> *v_geom_ID_face_1,
	std::vector<int> *v_geom_ID_face_2,
	std::vector<int> *v_geom_ID_face_3,
	int N, double cx, double cy, double R
)
{
	double pi = M_PI;
	double denom = (double)N-1.0;
	int N_nodes_curr = (*v_geom_f_node_X).size();
	
	for (int j = 0; j < N; j++)
	{
		double t_j = (2*pi/denom)*j;
		double x_j = R*cos(t_j) + cx;
		double y_j = R*sin(t_j) + cy;
		double z_j = 0.0;
		(*v_geom_f_node_X).push_back(x_j);
		(*v_geom_f_node_Y).push_back(y_j);
		(*v_geom_f_node_Z).push_back(z_j);
		
		if (j < N-1)
		{
			(*v_geom_ID_face_1).push_back( N_nodes_curr + j );
			(*v_geom_ID_face_2).push_back( N_nodes_curr + j+1 );
			(*v_geom_ID_face_3).push_back( 0 );
		}
	}
	
	return 0;
}

int AddPrism
(
	std::vector<double> *v_geom_f_node_X,
	std::vector<double> *v_geom_f_node_Y,
	std::vector<double> *v_geom_f_node_Z,
	std::vector<int> *v_geom_ID_face_1,
	std::vector<int> *v_geom_ID_face_2,
	std::vector<int> *v_geom_ID_face_3,
	double ax, double bx, double ay, double by, double az, double bz
)
{
	int N_nodes_curr = (*v_geom_f_node_X).size();
	
	// Nodes.
	(*v_geom_f_node_X).push_back(ax); // 0
	(*v_geom_f_node_Y).push_back(ay);
	(*v_geom_f_node_Z).push_back(az);
	(*v_geom_f_node_X).push_back(ax); // 1
	(*v_geom_f_node_Y).push_back(by);
	(*v_geom_f_node_Z).push_back(az);
	(*v_geom_f_node_X).push_back(ax); // 2
	(*v_geom_f_node_Y).push_back(ay);
	(*v_geom_f_node_Z).push_back(bz);
	(*v_geom_f_node_X).push_back(ax); // 3
	(*v_geom_f_node_Y).push_back(by);
	(*v_geom_f_node_Z).push_back(bz);
	(*v_geom_f_node_X).push_back(bx); // 4
	(*v_geom_f_node_Y).push_back(ay);
	(*v_geom_f_node_Z).push_back(az);
	(*v_geom_f_node_X).push_back(bx); // 5
	(*v_geom_f_node_Y).push_back(by);
	(*v_geom_f_node_Z).push_back(az);
	(*v_geom_f_node_X).push_back(bx); // 6
	(*v_geom_f_node_Y).push_back(ay);
	(*v_geom_f_node_Z).push_back(bz);
	(*v_geom_f_node_X).push_back(bx); // 7
	(*v_geom_f_node_Y).push_back(by);
	(*v_geom_f_node_Z).push_back(bz);
	
	// -X
	(*v_geom_ID_face_1).push_back( N_nodes_curr + 0 );
	(*v_geom_ID_face_2).push_back( N_nodes_curr + 2 );
	(*v_geom_ID_face_3).push_back( N_nodes_curr + 3 );
	(*v_geom_ID_face_1).push_back( N_nodes_curr + 0 );
	(*v_geom_ID_face_2).push_back( N_nodes_curr + 3 );
	(*v_geom_ID_face_3).push_back( N_nodes_curr + 1 );
	
	// +X
	(*v_geom_ID_face_1).push_back( N_nodes_curr + 4 );
	(*v_geom_ID_face_2).push_back( N_nodes_curr + 5 );
	(*v_geom_ID_face_3).push_back( N_nodes_curr + 7 );
	(*v_geom_ID_face_1).push_back( N_nodes_curr + 4 );
	(*v_geom_ID_face_2).push_back( N_nodes_curr + 7 );
	(*v_geom_ID_face_3).push_back( N_nodes_curr + 6 );
	
	// -Y
	(*v_geom_ID_face_1).push_back( N_nodes_curr + 0 );
	(*v_geom_ID_face_2).push_back( N_nodes_curr + 4 );
	(*v_geom_ID_face_3).push_back( N_nodes_curr + 6 );
	(*v_geom_ID_face_1).push_back( N_nodes_curr + 0 );
	(*v_geom_ID_face_2).push_back( N_nodes_curr + 6 );
	(*v_geom_ID_face_3).push_back( N_nodes_curr + 2 );
	
	// +Y
	(*v_geom_ID_face_1).push_back( N_nodes_curr + 1 );
	(*v_geom_ID_face_2).push_back( N_nodes_curr + 3 );
	(*v_geom_ID_face_3).push_back( N_nodes_curr + 7 );
	(*v_geom_ID_face_1).push_back( N_nodes_curr + 1 );
	(*v_geom_ID_face_2).push_back( N_nodes_curr + 7 );
	(*v_geom_ID_face_3).push_back( N_nodes_curr + 5 );
		
	// -Z
	(*v_geom_ID_face_1).push_back( N_nodes_curr + 7 );
	(*v_geom_ID_face_2).push_back( N_nodes_curr + 3 );
	(*v_geom_ID_face_3).push_back( N_nodes_curr + 2 );
	(*v_geom_ID_face_1).push_back( N_nodes_curr + 7 );
	(*v_geom_ID_face_2).push_back( N_nodes_curr + 2 );
	(*v_geom_ID_face_3).push_back( N_nodes_curr + 6 );
	
	// +Z
	(*v_geom_ID_face_1).push_back( N_nodes_curr + 0 );
	(*v_geom_ID_face_2).push_back( N_nodes_curr + 1 );
	(*v_geom_ID_face_3).push_back( N_nodes_curr + 5 );
	(*v_geom_ID_face_1).push_back( N_nodes_curr + 0 );
	(*v_geom_ID_face_2).push_back( N_nodes_curr + 5 );
	(*v_geom_ID_face_3).push_back( N_nodes_curr + 4 );
	
	return 0;
}

int AddSphere
(
	std::vector<double> *v_geom_f_node_X,
	std::vector<double> *v_geom_f_node_Y,
	std::vector<double> *v_geom_f_node_Z,
	std::vector<int> *v_geom_ID_face_1,
	std::vector<int> *v_geom_ID_face_2,
	std::vector<int> *v_geom_ID_face_3,
	int N1, int N2, double cx, double cy, double cz, double R
)
{
	double pi = M_PI;
	double denom1 = (double)N1-1.0;
	double denom2 = (double)N2-1.0;
	int N_nodes_curr = (*v_geom_f_node_X).size();
	
	// N1 and N2 are resolutions in the polar (0-pi) and azimuthal (0-2*pi) axes, respectively.
	for (int j = 0; j < N1; j++)
	{
		for (int k = 0; k < N2; k++)
		{
			double t_jk = (pi/denom1)*j;
			double p_jk = (2*pi/denom2)*k;
			double x_jk = R*sin(t_jk)*cos(p_jk) + cx;
			double y_jk = R*sin(t_jk)*sin(p_jk) + cy;
			double z_jk = R*cos(t_jk) + cz;
			(*v_geom_f_node_X).push_back(x_jk);
			(*v_geom_f_node_Y).push_back(y_jk);
			(*v_geom_f_node_Z).push_back(z_jk);
			
			if (j < N1-1 && k < N2-1)
			{
				(*v_geom_ID_face_1).push_back( N_nodes_curr + ((k+0)+N2*(j+0)) );
				(*v_geom_ID_face_2).push_back( N_nodes_curr + ((k+1)+N2*(j+0)) );
				(*v_geom_ID_face_3).push_back( N_nodes_curr + ((k+1)+N2*(j+1)) );
				
				(*v_geom_ID_face_1).push_back( N_nodes_curr + ((k+0)+N2*(j+0)) );
				(*v_geom_ID_face_2).push_back( N_nodes_curr + ((k+1)+N2*(j+1)) );
				(*v_geom_ID_face_3).push_back( N_nodes_curr + ((k+0)+N2*(j+1)) );
			}
		}
	}
	
	return 0;
}

int AddNACA002D
(
	std::vector<double> *v_geom_f_node_X,
	std::vector<double> *v_geom_f_node_Y,
	std::vector<double> *v_geom_f_node_Z,
	std::vector<int> *v_geom_ID_face_1,
	std::vector<int> *v_geom_ID_face_2,
	std::vector<int> *v_geom_ID_face_3,
	int N, double t, double ax, double bx, double ay, double by, int te
)
{
	double denom = (double)N-1.0;
	double last_coeff = -0.1036;
	if (te)
		last_coeff = -0.1015;
	int N_nodes_curr = (*v_geom_f_node_X).size();
	
	// Nodes.
	for (int j = 0; j < N; j++)
	{
		double x_j = ax + ((bx-ax)/denom)*j;
		double x_jp = (1.0/denom)*j;
		double y_j = (ay+by)/2.0 + 5*t*(0.2969*sqrt(x_jp) + x_jp*(-0.1260 + x_jp*(-0.3516 + x_jp*(0.2843 + x_jp*(last_coeff)))));
		double z_j = 0.0;
		(*v_geom_f_node_X).push_back(x_j);
		(*v_geom_f_node_Y).push_back(y_j);
		(*v_geom_f_node_Z).push_back(z_j);
	}
	for (int j = 0; j < N; j++)
	{
		double x_j = ax + ((bx-ax)/denom)*j;
		double x_jp = (1.0/denom)*j;
		double y_j = (ay+by)/2.0 + -5*t*(0.2969*sqrt(x_jp) + x_jp*(-0.1260 + x_jp*(-0.3516 + x_jp*(0.2843 + x_jp*(last_coeff)))));
		double z_j = 0.0;
		(*v_geom_f_node_X).push_back(x_j);
		(*v_geom_f_node_Y).push_back(y_j);
		(*v_geom_f_node_Z).push_back(z_j);
	}
	
	// Upper edge.
	for (int j = N-1; j > 0; j--)
	{
		(*v_geom_ID_face_1).push_back( N_nodes_curr + 0+(j+0) );
		(*v_geom_ID_face_2).push_back( N_nodes_curr + 0+(j-1) );
		(*v_geom_ID_face_3).push_back( 0 );
	}
	
	// Lower edge.
	for (int j = 0; j < N-1; j++)
	{
		(*v_geom_ID_face_1).push_back( N_nodes_curr + N+(j+0) );
		(*v_geom_ID_face_2).push_back( N_nodes_curr + N+(j+1) );
		(*v_geom_ID_face_3).push_back( 0 );
	}
	
	// Trailing edge, if applicable.
	if (te)
	{
		(*v_geom_ID_face_1).push_back( N_nodes_curr + 2*N-1 );
		(*v_geom_ID_face_2).push_back( N_nodes_curr + N-1 );
		(*v_geom_ID_face_3).push_back( 0 );
	}
	
	return 0;
}

int AddNACA003D
(
	std::vector<double> *v_geom_f_node_X,
	std::vector<double> *v_geom_f_node_Y,
	std::vector<double> *v_geom_f_node_Z,
	std::vector<int> *v_geom_ID_face_1,
	std::vector<int> *v_geom_ID_face_2,
	std::vector<int> *v_geom_ID_face_3,
	int N, double t, double ax, double bx, double ay, double by, double az, double bz, int te
)
{
	double denom = (double)N-1.0;
	double last_coeff = -0.1036;
	if (te)
		last_coeff = -0.1015;
	int N_nodes_curr = (*v_geom_f_node_X).size();
	
	// Nodes.
	for (int j = 0; j < N; j++)
	{
		double x_j = ax + ((bx-ax)/denom)*j;
		double y_j = by;
		double z_j = (bz+az)/2.0;
		(*v_geom_f_node_X).push_back(x_j);
		(*v_geom_f_node_Y).push_back(y_j);
		(*v_geom_f_node_Z).push_back(z_j);
	}
	for (int j = 0; j < N; j++)
	{
		double x_j = ax + ((bx-ax)/denom)*j;
		double x_jp = (1.0/denom)*j;
		double y_j = by;
		//double z_j = bz;
		double z_j = (bz+az)/2.0 + 5*t*(0.2969*sqrt(x_jp) + x_jp*(-0.1260 + x_jp*(-0.3516 + x_jp*(0.2843 + x_jp*(last_coeff)))));
		(*v_geom_f_node_X).push_back(x_j);
		(*v_geom_f_node_Y).push_back(y_j);
		(*v_geom_f_node_Z).push_back(z_j);
	}
	for (int j = 0; j < N; j++)
	{
		double x_j = ax + ((bx-ax)/denom)*j;
		double x_jp = (1.0/denom)*j;
		double y_j = by;
		//double z_j = az;
		double z_j = (bz+az)/2.0 + -5*t*(0.2969*sqrt(x_jp) + x_jp*(-0.1260 + x_jp*(-0.3516 + x_jp*(0.2843 + x_jp*(last_coeff)))));
		(*v_geom_f_node_X).push_back(x_j);
		(*v_geom_f_node_Y).push_back(y_j);
		(*v_geom_f_node_Z).push_back(z_j);
	}
	for (int j = 0; j < N; j++)
	{
		double x_j = ax + ((bx-ax)/denom)*j;
		double y_j = ay;
		double z_j = (bz+az)/2.0;
		(*v_geom_f_node_X).push_back(x_j);
		(*v_geom_f_node_Y).push_back(y_j);
		(*v_geom_f_node_Z).push_back(z_j);
	}
	for (int j = 0; j < N; j++)
	{
		double x_j = ax + ((bx-ax)/denom)*j;
		double x_jp = (1.0/denom)*j;
		double y_j = ay;
		//double z_j = bz;
		double z_j = (bz+az)/2.0 + 5*t*(0.2969*sqrt(x_jp) + x_jp*(-0.1260 + x_jp*(-0.3516 + x_jp*(0.2843 + x_jp*(last_coeff)))));
		(*v_geom_f_node_X).push_back(x_j);
		(*v_geom_f_node_Y).push_back(y_j);
		(*v_geom_f_node_Z).push_back(z_j);
	}
	for (int j = 0; j < N; j++)
	{
		double x_j = ax + ((bx-ax)/denom)*j;
		double x_jp = (1.0/denom)*j;
		double y_j = ay;
		//double z_j = az;
		double z_j = (bz+az)/2.0 + -5*t*(0.2969*sqrt(x_jp) + x_jp*(-0.1260 + x_jp*(-0.3516 + x_jp*(0.2843 + x_jp*(last_coeff)))));
		(*v_geom_f_node_X).push_back(x_j);
		(*v_geom_f_node_Y).push_back(y_j);
		(*v_geom_f_node_Z).push_back(z_j);
	}
	
	// Right side.
	for (int j = 0; j < N-1; j++)
	{
		// Upper.
		(*v_geom_ID_face_1).push_back( N_nodes_curr + ((j+0)+N*(0)) );
		(*v_geom_ID_face_2).push_back( N_nodes_curr + ((j+1)+N*(0)) );
		(*v_geom_ID_face_3).push_back( N_nodes_curr + ((j+1)+N*(1)) );
		(*v_geom_ID_face_1).push_back( N_nodes_curr + ((j+0)+N*(0)) );
		(*v_geom_ID_face_2).push_back( N_nodes_curr + ((j+1)+N*(1)) );
		(*v_geom_ID_face_3).push_back( N_nodes_curr + ((j+0)+N*(1)) );
		
		// Lower.
		(*v_geom_ID_face_1).push_back( N_nodes_curr + ((j+0)+N*(0)) );
		(*v_geom_ID_face_2).push_back( N_nodes_curr + ((j+1)+N*(2)) );
		(*v_geom_ID_face_3).push_back( N_nodes_curr + ((j+1)+N*(0)) );
		(*v_geom_ID_face_1).push_back( N_nodes_curr + ((j+0)+N*(0)) );
		(*v_geom_ID_face_2).push_back( N_nodes_curr + ((j+0)+N*(2)) );
		(*v_geom_ID_face_3).push_back( N_nodes_curr + ((j+1)+N*(2)) );
	}
	
	// Left side.
	for (int j = 0; j < N-1; j++)
	{
		// Upper.
		(*v_geom_ID_face_1).push_back( N_nodes_curr + ((j+0)+N*(3)) );
		(*v_geom_ID_face_2).push_back( N_nodes_curr + ((j+1)+N*(4)) );
		(*v_geom_ID_face_3).push_back( N_nodes_curr + ((j+1)+N*(3)) );
		(*v_geom_ID_face_1).push_back( N_nodes_curr + ((j+0)+N*(3)) );
		(*v_geom_ID_face_2).push_back( N_nodes_curr + ((j+0)+N*(4)) );
		(*v_geom_ID_face_3).push_back( N_nodes_curr + ((j+1)+N*(4)) );
		
		// Lower.
		(*v_geom_ID_face_1).push_back( N_nodes_curr + ((j+0)+N*(3)) );
		(*v_geom_ID_face_2).push_back( N_nodes_curr + ((j+1)+N*(3)) );
		(*v_geom_ID_face_3).push_back( N_nodes_curr + ((j+1)+N*(5)) );
		(*v_geom_ID_face_1).push_back( N_nodes_curr + ((j+0)+N*(3)) );
		(*v_geom_ID_face_2).push_back( N_nodes_curr + ((j+1)+N*(5)) );
		(*v_geom_ID_face_3).push_back( N_nodes_curr + ((j+0)+N*(5)) );
	}
	
	// Top and bottom sides.
	for (int j = 0; j < N-1; j++)
	{
		// Upper.
		(*v_geom_ID_face_1).push_back( N_nodes_curr + ((j+0)+N*(1)) );
		(*v_geom_ID_face_2).push_back( N_nodes_curr + ((j+1)+N*(1)) );
		(*v_geom_ID_face_3).push_back( N_nodes_curr + ((j+1)+N*(4)) );
		(*v_geom_ID_face_1).push_back( N_nodes_curr + ((j+0)+N*(1)) );
		(*v_geom_ID_face_2).push_back( N_nodes_curr + ((j+1)+N*(4)) );
		(*v_geom_ID_face_3).push_back( N_nodes_curr + ((j+0)+N*(4)) );
		
		// Lower.
		(*v_geom_ID_face_1).push_back( N_nodes_curr + ((j+0)+N*(5)) );
		(*v_geom_ID_face_2).push_back( N_nodes_curr + ((j+1)+N*(5)) );
		(*v_geom_ID_face_3).push_back( N_nodes_curr + ((j+1)+N*(2)) );
		(*v_geom_ID_face_1).push_back( N_nodes_curr + ((j+0)+N*(5)) );
		(*v_geom_ID_face_2).push_back( N_nodes_curr + ((j+1)+N*(2)) );
		(*v_geom_ID_face_3).push_back( N_nodes_curr + ((j+0)+N*(2)) );
	}
	
	// Trailing edge, if applicable.
	if (te)
	{
		(*v_geom_ID_face_1).push_back( N_nodes_curr + ((N-1)+N*(0)) );
		(*v_geom_ID_face_2).push_back( N_nodes_curr + ((N-1)+N*(3)) );
		(*v_geom_ID_face_3).push_back( N_nodes_curr + ((N-1)+N*(4)) );
		(*v_geom_ID_face_1).push_back( N_nodes_curr + ((N-1)+N*(0)) );
		(*v_geom_ID_face_2).push_back( N_nodes_curr + ((N-1)+N*(4)) );
		(*v_geom_ID_face_3).push_back( N_nodes_curr + ((N-1)+N*(1)) );
		(*v_geom_ID_face_1).push_back( N_nodes_curr + ((N-1)+N*(2)) );
		(*v_geom_ID_face_2).push_back( N_nodes_curr + ((N-1)+N*(5)) );
		(*v_geom_ID_face_3).push_back( N_nodes_curr + ((N-1)+N*(3)) );
		(*v_geom_ID_face_1).push_back( N_nodes_curr + ((N-1)+N*(2)) );
		(*v_geom_ID_face_2).push_back( N_nodes_curr + ((N-1)+N*(3)) );
		(*v_geom_ID_face_3).push_back( N_nodes_curr + ((N-1)+N*(0)) );
	}
	
	return 0;
}

int PrintSTL
(
	std::string output_dir,
	std::vector<double> *v_geom_f_node_X,
	std::vector<double> *v_geom_f_node_Y,
	std::vector<double> *v_geom_f_node_Z,
	std::vector<int> *v_geom_ID_face_1,
	std::vector<int> *v_geom_ID_face_2,
	std::vector<int> *v_geom_ID_face_3
)
{
	std::ofstream stl = std::ofstream(output_dir + "geometry.stl");
	stl << "solid Geometry" << std::endl;
	
	// Compute facet normals and print.
	for (int i = 0; i < (*v_geom_ID_face_1).size(); i++)
	{
		int p1 = (*v_geom_ID_face_1)[i];
		int p2 = (*v_geom_ID_face_2)[i];
		int p3 = (*v_geom_ID_face_3)[i];
		double v1x = (*v_geom_f_node_X)[p1];
		double v1y = (*v_geom_f_node_Y)[p1];
		double v1z = (*v_geom_f_node_Z)[p1];
		double v2x = (*v_geom_f_node_X)[p2];
		double v2y = (*v_geom_f_node_Y)[p2];
		double v2z = (*v_geom_f_node_Z)[p2];
		double v3x = (*v_geom_f_node_X)[p3];
		double v3y = (*v_geom_f_node_Y)[p3];
		double v3z = (*v_geom_f_node_Z)[p3];
#if (N_DIM==2)
		double n1 = v2y-v1y;
		double n2 = -(v2x-v1x);
		double n3 = 0.0;
#else
		double dx1 = v2x-v1x;
		double dy1 = v2y-v1y;
		double dz1 = v2z-v1z;
		double dx2 = v3x-v1x;
		double dy2 = v3y-v1y;
		double dz2 = v3z-v1z;
		double n1 = dy1*dz2-dz1*dy2;
		double n2 = dz1*dx2-dx1*dz2;
		double n3 = dx1*dy2-dy1*dx2;
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
	std::vector<double> *v_geom_f_node_X,
	std::vector<double> *v_geom_f_node_Y,
	std::vector<double> *v_geom_f_node_Z,
	std::vector<int> *v_geom_ID_face_1,
	std::vector<int> *v_geom_ID_face_2,
	std::vector<int> *v_geom_ID_face_3
)
{
	std::ofstream obj = std::ofstream(output_dir + "geometry.obj");
	obj << "# Geometry." << std::endl << std::endl;;
	
	// Print vertices.
	obj << "Vertices." << std::endl;
	for (int i = 0; i < (*v_geom_f_node_X).size(); i++)
		obj << "v " << (*v_geom_f_node_X)[i] << " " << (*v_geom_f_node_Y)[i] << " " << (*v_geom_f_node_Z)[i] << std::endl;
	obj << std::endl;
	
	// Print facs.
	obj << "Faces." << std::endl;
	for (int i = 0; i < (*v_geom_ID_face_1).size(); i++)
		obj << "f " << (*v_geom_ID_face_1)[i]+1 << " " << (*v_geom_ID_face_2)[i]+1 << " " << (*v_geom_ID_face_3)[i]+1 << std::endl;
	
	obj.close();

	return 0;
}

int Geometry::G_ImportBoundariesFromTextFile(int i_dev)
{
	std::vector<double> v_geom_f_node_X;
	std::vector<double> v_geom_f_node_Y;
	std::vector<double> v_geom_f_node_Z;
	std::vector<int> v_geom_ID_face_1;
	std::vector<int> v_geom_ID_face_2;
	std::vector<int> v_geom_ID_face_3;
	
	// Add shapes defined in input file as necessary.
	std::cout << "[-] Reading geometry from input..." << std::endl;
	std::string word = "";
	std::ifstream input_geometry = std::ifstream("../input/geometry.txt");
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
					AddBoundingBox(
						&v_geom_f_node_X,
						&v_geom_f_node_Y,
						&v_geom_f_node_Z,
						&v_geom_ID_face_1,
						&v_geom_ID_face_2,
						&v_geom_ID_face_3,
						0, Lx, 0, Ly, 0, Lz
					);
					std::cout << "Detected BOX, adding..." << std::endl;
				}
				if (word == "RECTANGLE")
				{
					std::string s;
					double ax, bx, ay, by;
					ss >> s;
					if (s == "CORNER")
						ss >> ax >> bx >> ay >> by;
					else
					{
						double cx, cy, lx, ly;
						ss >> cx >> cy >> lx >> ly;
						ax = cx-lx/2.0;
						bx = cx+lx/2.0;
						ay = cy-ly/2.0;
						by = cy+ly/2.0;
					}
					
					AddRectangle(
						&v_geom_f_node_X,
						&v_geom_f_node_Y,
						&v_geom_f_node_Z,
						&v_geom_ID_face_1,
						&v_geom_ID_face_2,
						&v_geom_ID_face_3,
						ax, bx, ay, by
					);
				}
				if (word == "CIRCLE")
				{
					int N;
					double cx, cy, R;
					ss >> N >> cx >> cy >> R;
					AddCircle(
						&v_geom_f_node_X,
						&v_geom_f_node_Y,
						&v_geom_f_node_Z,
						&v_geom_ID_face_1,
						&v_geom_ID_face_2,
						&v_geom_ID_face_3,
						N, cx, cy, R
					);
				}
				if (word == "PRISM")
				{
					std::string s;
					double ax, bx, ay, by, az, bz;
					ss >> s;
					if (s == "CORNER")
						ss >> ax >> bx >> ay >> by >> az >> bz;
					else
					{
						double cx, cy, cz, lx, ly, lz;
						ss >> cx >> cy >> cz >> lx >> ly >> lz;
						ax = cx-lx/2.0;
						bx = cx+lx/2.0;
						ay = cy-ly/2.0;
						by = cy+ly/2.0;
						az = cz-lz/2.0;
						bz = cz+lz/2.0;
					}
					
					AddPrism(
						&v_geom_f_node_X,
						&v_geom_f_node_Y,
						&v_geom_f_node_Z,
						&v_geom_ID_face_1,
						&v_geom_ID_face_2,
						&v_geom_ID_face_3,
						ax, bx, ay, by, az, bz
					);
				}
				if (word == "SPHERE")
				{
					int N1, N2;
					double cx, cy, cz, R;
					ss >> N1 >> N2 >> cx >> cy >> cz >> R;
					AddSphere(
						&v_geom_f_node_X,
						&v_geom_f_node_Y,
						&v_geom_f_node_Z,
						&v_geom_ID_face_1,
						&v_geom_ID_face_2,
						&v_geom_ID_face_3,
						N1, N2, cx, cy, cz, R
					);
				}
				if (word == "NACA002D")
				{
					std::string s;
					int N, te;
					double t, ax, bx, ay, by;
					ss >> s;
					if (s == "CORNER")
						ss >> N >> t >> ax >> bx >> ay >> by >> te;
					else
					{
						double cx, cy, lx, ly;
						ss >> N >> t >> cx >> cy >> lx >> ly >> te;
						ax = cx-lx/2.0;
						bx = cx+lx/2.0;
						ay = cy-ly/2.0;
						by = cy+ly/2.0;
					}
					
					AddNACA002D(
						&v_geom_f_node_X,
						&v_geom_f_node_Y,
						&v_geom_f_node_Z,
						&v_geom_ID_face_1,
						&v_geom_ID_face_2,
						&v_geom_ID_face_3,
						N, t*0.01*(bx-ax), ax, bx, ay, by, te
					);
				}
				if (word == "NACA003D")
				{
					std::string s;
					int N, te;
					double t, ax, bx, ay, by, az, bz;
					ss >> s;
					if (s == "CORNER")
						ss >> N >> t >> ax >> bx >> ay >> by >> az >> bz >> te;
					else
					{
						double cx, cy, cz, lx, ly, lz;
						ss >> N >> t >> cx >> cy >> cz >> lx >> ly >> lz >> te;
						ax = cx-lx/2.0;
						bx = cx+lx/2.0;
						ay = cy-ly/2.0;
						by = cy+ly/2.0;
						az = cz-lz/2.0;
						bz = cz+lz/2.0;
					}
					
					AddNACA003D(
						&v_geom_f_node_X,
						&v_geom_f_node_Y,
						&v_geom_f_node_Z,
						&v_geom_ID_face_1,
						&v_geom_ID_face_2,
						&v_geom_ID_face_3,
						N, t*0.01*(bx-ax), ax, bx, ay, by, az, bz, te
					);
				}
			}
		}
	}
	
	// Make node and face indices permanent. 
	n_nodes[i_dev] = v_geom_f_node_X.size();
	n_faces[i_dev] = v_geom_ID_face_1.size();
	n_nodes[i_dev] = n_nodes[i_dev] + 32-(n_nodes[i_dev]%32);
	n_faces[i_dev] = n_faces[i_dev] + 32-(n_faces[i_dev]%32);
	geom_f_node_X[i_dev] = new double[3*n_nodes[i_dev]];
	geom_ID_face[i_dev] = new int[3*n_faces[i_dev]];
	geom_ID_face_attr[i_dev] = new double[n_faces[i_dev]];
	for (int j = 0; j < n_nodes[i_dev]; j++)
	{
		geom_f_node_X[i_dev][j + 0*n_nodes[i_dev]] = 0.0;
		geom_f_node_X[i_dev][j + 1*n_nodes[i_dev]] = 0.0;
		geom_f_node_X[i_dev][j + 2*n_nodes[i_dev]] = 0.0;
		if (j < v_geom_f_node_X.size())
		{
			geom_f_node_X[i_dev][j + 0*n_nodes[i_dev]] = v_geom_f_node_X[j];
			geom_f_node_X[i_dev][j + 1*n_nodes[i_dev]] = v_geom_f_node_Y[j];
			geom_f_node_X[i_dev][j + 2*n_nodes[i_dev]] = v_geom_f_node_Z[j];
		}
	}
	for (int j = 0; j < n_faces[i_dev]; j++)
	{
		geom_ID_face[i_dev][j + 0*n_faces[i_dev]] = -1;
		geom_ID_face[i_dev][j + 1*n_faces[i_dev]] = -1;
		geom_ID_face[i_dev][j + 2*n_faces[i_dev]] = -1;
		if (j < v_geom_ID_face_1.size())
		{
			geom_ID_face[i_dev][j + 0*n_faces[i_dev]] = v_geom_ID_face_1[j];
			geom_ID_face[i_dev][j + 1*n_faces[i_dev]] = v_geom_ID_face_2[j];
			geom_ID_face[i_dev][j + 2*n_faces[i_dev]] = v_geom_ID_face_3[j];
		}
	}
	
	
	// Print the geometry to an STL file to superimpose over the computational grid in Paraview.
	std::cout << "[-] Finished reading, print STL file..." << std::endl;
// 	PrintOBJ(
// 		output_dir,
// 		&v_geom_f_node_X,
// 		&v_geom_f_node_Y,
// 		&v_geom_f_node_Z,
// 		&v_geom_ID_face_1,
// 		&v_geom_ID_face_2,
// 		&v_geom_ID_face_3
// 	);
	PrintOBJ(i_dev);
	std::cout << "[-] Finished printing STL file..." << std::endl;
	
	return 0;
}

/*
int Mesh::M_CheckFaceIntersection(int i_dev, int i, int j, int k, int i_face)
{
	double x1,x2,x3,x4;
	double y1,y2,y3,y4;
	double x_bi_0 = (i-1)*dx_cblock;
	double y_bi_0 = (j-1)*dx_cblock;
	double x_bi_1 = (i-1)*dx_cblock + dx_cblock;
	double y_bi_1 = (j-1)*dx_cblock + dx_cblock;
	double x_f_0 = faces_f_X[i_dev][i_face];
	double y_f_1 = 
	
#if (N_DIM==2)
	// Calculation source: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line
	x1 = x_bi_0; y1 = y_bi_0;
	x2 = x_bi_1; y2 = y_bi_0;
	x3 = 
	double t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4));
	double u = -((x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)) / ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4));
#else
	double z_bi_0 = (k-1)*dx_cblock;
	double z_bi_1 = (k-1)*dx_cblock + dx_cblock;
#endif
	
	return 0;
}
*/
