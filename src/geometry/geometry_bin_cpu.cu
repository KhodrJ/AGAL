/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#include "geometry.h"

template <typename ufloat_g_t, int N_DIM>
inline bool IncludeInBin
(
	const ufloat_g_t xm,
	const ufloat_g_t xM,
	const ufloat_g_t ym,
	const ufloat_g_t yM,
	const ufloat_g_t zm,
	const ufloat_g_t zM,
	const ufloat_g_t vBx_m,
	const ufloat_g_t vBx_M,
	const ufloat_g_t vBy_m,
	const ufloat_g_t vBy_M,
	const ufloat_g_t vBz_m,
	const ufloat_g_t vBz_M,
	const ufloat_g_t vx1,
	const ufloat_g_t vy1,
	const ufloat_g_t vz1,
	const ufloat_g_t vx2,
	const ufloat_g_t vy2,
	const ufloat_g_t vz2,
	const ufloat_g_t vx3,
	const ufloat_g_t vy3,
	const ufloat_g_t vz3
)
{
	ufloat_g_t tmp = (ufloat_g_t)(0.0);
	ufloat_g_t ex1 = (ufloat_g_t)(0.0);
	ufloat_g_t ey1 = (ufloat_g_t)(0.0);
	
if (N_DIM==2)
{
	// Only consider this calculation if the bounding box intersects the bin.
	if ( !( (vBx_m < xm && vBx_M < xm) || (vBx_m > xM && vBx_M > xM) || (vBy_m < ym && vBy_M < ym) || (vBy_m > yM && vBy_M > yM) ) )
	{
		// Check if bounding box is entirely inside current bin.
		if (vBx_m > xm && vBx_M < xM && vBy_m > ym && vBy_M < yM) { return true; }
		
		// Check if at least one of the vertices is inside the bin.
		if (vx1 > xm && vx1 < xM) { return true; }
		if (vx2 > xm && vx2 < xM) { return true; }
		if (vy1 > ym && vy1 < yM) { return true; }
		if (vy2 > ym && vy2 < yM) { return true; }
		
		// Check the bottom edge of the bin.
		{
			ey1 = vy2-vy1;
			tmp = (ym-vy1)/(ey1);
			ex1 = vx1 + tmp*(vx2-vx1);
			ey1 = vy1 + tmp*(vy2-vy1);
			if (CheckInLine(tmp,ex1,xm,xM)) { return true; }
		}
		
		// Check the top edge of the bin.
		{
			ey1 = vy2-vy1;
			tmp = (yM-vy1)/(ey1);
			ex1 = vx1 + tmp*(vx2-vx1);
			ey1 = vy1 + tmp*(vy2-vy1);
			if (CheckInLine(tmp,ex1,xm,xM)) { return true; }
		}
		
		// Check the left edge of the bin.
		{
			ex1 = vx2-vx1;
			tmp = (xm-vx1)/(ex1);
			ex1 = vx1 + tmp*(vx2-vx1);
			ey1 = vy1 + tmp*(vy2-vy1);
			if (CheckInLine(tmp,ey1,ym,yM)) { return true; }
		}
		
		// Check the right edge of the bin.
		{
			ex1 = vx2-vx1;
			tmp = (xM-vx1)/(ex1);
			ex1 = vx1 + tmp*(vx2-vx1);
			ey1 = vy1 + tmp*(vy2-vy1);
			if (CheckInLine(tmp,ey1,ym,yM)) { return true; }
		}
	}
}
else
{
	ufloat_g_t ez1 = (ufloat_g_t)(0.0);
	
	if ( !( (vBx_m < xm && vBx_M < xm) || (vBx_m > xM && vBx_M > xM) || (vBy_m < ym && vBy_M < ym) || (vBy_m > yM && vBy_M > yM) || (vBz_m < zm && vBz_M < zm) || (vBz_m > zM && vBz_M > zM) ) )
	{
		// Check if bounding box is entirely inside current bin.
		if (vBx_m > xm && vBx_M < xM && vBy_m > ym && vBy_M < yM && vBz_m > zm && vBz_M < zM) { return true; }
		
		// Check if bounding box completely surrounds the bin.
		if (vBx_m < xm && vBx_M > xM && vBy_m < ym && vBy_M > yM) { return true; }
		if (vBy_m < ym && vBy_M > yM && vBz_m < zm && vBz_M > zM) { return true; }
		if (vBz_m < zm && vBz_M > zM && vBx_m < xm && vBx_M > xM) { return true; }
		
		// Check if at least one of the vertices is inside the bin.
		if (vx1 > xm && vx1 < xM && vy1 > ym && vy1 < yM && vz1 > zm && vz1 < zM) { return true; }
		if (vx2 > xm && vx2 < xM && vy2 > ym && vy2 < yM && vz2 > zm && vz2 < zM) { return true; }
		if (vx3 > xm && vx3 < xM && vy3 > ym && vy3 < yM && vz3 > zm && vz3 < zM) { return true; }
		
		// Check the bottom face of the bin.
		{
			ez1 = vz2-vz1;
			tmp = (zm-vz1)/(ez1);
			ex1 = vx1 + tmp*(vx2-vx1);
			ey1 = vy1 + tmp*(vy2-vy1);
			ez1 = vz1 + tmp*(vz2-vz1);
			if (CheckInRect(tmp,ex1,ey1,xm,ym,xM,yM)) { return true; }
		}
		{
			ez1 = vz3-vz2;
			tmp = (zm-vz2)/(ez1);
			ex1 = vx2 + tmp*(vx3-vx2);
			ey1 = vy2 + tmp*(vy3-vy2);
			ez1 = vz2 + tmp*(vz3-vz2);
			if (CheckInRect(tmp,ex1,ey1,xm,ym,xM,yM)) { return true; }
		}
		{
			ez1 = vz1-vz3;
			tmp = (zm-vz3)/(ez1);
			ex1 = vx3 + tmp*(vx1-vx3);
			ey1 = vy3 + tmp*(vy1-vy3);
			ez1 = vz3 + tmp*(vz1-vz3);
			if (CheckInRect(tmp,ex1,ey1,xm,ym,xM,yM)) { return true; }
		}
		
		// Check the top face of the bin.
		{
			ez1 = vz2-vz1;
			tmp = (zM-vz1)/(ez1);
			ex1 = vx1 + tmp*(vx2-vx1);
			ey1 = vy1 + tmp*(vy2-vy1);
			ez1 = vz1 + tmp*(vz2-vz1);
			if (CheckInRect(tmp,ex1,ey1,xm,ym,xM,yM)) { return true; }
		}
		{
			ez1 = vz3-vz2;
			tmp = (zM-vz2)/(ez1);
			ex1 = vx2 + tmp*(vx3-vx2);
			ey1 = vy2 + tmp*(vy3-vy2);
			ez1 = vz2 + tmp*(vz3-vz2);
			if (CheckInRect(tmp,ex1,ey1,xm,ym,xM,yM)) { return true; }
		}
		{
			ez1 = vz1-vz3;
			tmp = (zM-vz3)/(ez1);
			ex1 = vx3 + tmp*(vx1-vx3);
			ey1 = vy3 + tmp*(vy1-vy3);
			ez1 = vz3 + tmp*(vz1-vz3);
			if (CheckInRect(tmp,ex1,ey1,xm,ym,xM,yM)) { return true; }
		}
		
		// Check the back face of the bin.
		{
			ey1 = vy2-vy1;
			tmp = (ym-vy1)/(ey1);
			ex1 = vx1 + tmp*(vx2-vx1);
			ey1 = vy1 + tmp*(vy2-vy1);
			ez1 = vz1 + tmp*(vz2-vz1);
			if (CheckInRect(tmp,ex1,ez1,xm,zm,xM,zM)) { return true; }
		}
		{
			ey1 = vy3-vy2;
			tmp = (ym-vy2)/(ey1);
			ex1 = vx2 + tmp*(vx3-vx2);
			ey1 = vy2 + tmp*(vy3-vy2);
			ez1 = vz2 + tmp*(vz3-vz2);
			if (CheckInRect(tmp,ex1,ez1,xm,zm,xM,zM)) { return true; }
		}
		{
			ey1 = vy1-vy3;
			tmp = (ym-vy3)/(ey1);
			ex1 = vx3 + tmp*(vx1-vx3);
			ey1 = vy3 + tmp*(vy1-vy3);
			ez1 = vz3 + tmp*(vz1-vz3);
			if (CheckInRect(tmp,ex1,ez1,xm,zm,xM,zM)) { return true; }
		}
		
		// Check the front face of the bin.
		{
			ey1 = vy2-vy1;
			tmp = (yM-vy1)/(ey1);
			ex1 = vx1 + tmp*(vx2-vx1);
			ey1 = vy1 + tmp*(vy2-vy1);
			ez1 = vz1 + tmp*(vz2-vz1);
			if (CheckInRect(tmp,ex1,ez1,xm,zm,xM,zM)) { return true; }
		}
		{
			ey1 = vy3-vy2;
			tmp = (yM-vy2)/(ey1);
			ex1 = vx2 + tmp*(vx3-vx2);
			ey1 = vy2 + tmp*(vy3-vy2);
			ez1 = vz2 + tmp*(vz3-vz2);
			if (CheckInRect(tmp,ex1,ez1,xm,zm,xM,zM)) { return true; }
		}
		{
			ey1 = vy1-vy3;
			tmp = (yM-vy3)/(ey1);
			ex1 = vx3 + tmp*(vx1-vx3);
			ey1 = vy3 + tmp*(vy1-vy3);
			ez1 = vz3 + tmp*(vz1-vz3);
			if (CheckInRect(tmp,ex1,ez1,xm,zm,xM,zM)) { return true; }
		}
		
		// Check the left face of the bin.
		{
			ex1 = vx2-vx1;
			tmp = (xm-vx1)/(ex1);
			ex1 = vx1 + tmp*(vx2-vx1);
			ey1 = vy1 + tmp*(vy2-vy1);
			ez1 = vz1 + tmp*(vz2-vz1);
			if (CheckInRect(tmp,ey1,ez1,ym,zm,yM,zM)) { return true; }
		}
		{
			ex1 = vx3-vx2;
			tmp = (xm-vx2)/(ex1);
			ex1 = vx2 + tmp*(vx3-vx2);
			ey1 = vy2 + tmp*(vy3-vy2);
			ez1 = vz2 + tmp*(vz3-vz2);
			if (CheckInRect(tmp,ey1,ez1,ym,zm,yM,zM)) { return true; }
		}
		{
			ex1 = vx1-vx3;
			tmp = (xm-vx3)/(ex1);
			ex1 = vx3 + tmp*(vx1-vx3);
			ey1 = vy3 + tmp*(vy1-vy3);
			ez1 = vz3 + tmp*(vz1-vz3);
			if (CheckInRect(tmp,ey1,ez1,ym,zm,yM,zM)) { return true; }
		}
		
		// Check the right face of the bin.
		{
			ex1 = vx2-vx1;
			tmp = (xM-vx1)/(ex1);
			ex1 = vx1 + tmp*(vx2-vx1);
			ey1 = vy1 + tmp*(vy2-vy1);
			ez1 = vz1 + tmp*(vz2-vz1);
			if (CheckInRect(tmp,ey1,ez1,ym,zm,yM,zM)) { return true; }
		}
		{
			ex1 = vx3-vx2;
			tmp = (xM-vx2)/(ex1);
			ex1 = vx2 + tmp*(vx3-vx2);
			ey1 = vy2 + tmp*(vy3-vy2);
			ez1 = vz2 + tmp*(vz3-vz2);
			if (CheckInRect(tmp,ey1,ez1,ym,zm,yM,zM)) { return true; }
		}
		{
			ex1 = vx1-vx3;
			tmp = (xM-vx3)/(ex1);
			ex1 = vx3 + tmp*(vx1-vx3);
			ey1 = vy3 + tmp*(vy1-vy3);
			ez1 = vz3 + tmp*(vz1-vz3);
			if (CheckInRect(tmp,ey1,ez1,ym,zm,yM,zM)) { return true; }
		}
	}
}
	
	return false;
}



template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP>
int Geometry<ufloat_t,ufloat_g_t,AP>::G_MakeBinsCPU(int i_dev)
{
	// Some constants.
	ufloat_g_t Lx0g __attribute__((unused)) = Lx/(ufloat_g_t)G_BIN_DENSITY;   // Get bin lengths along axes.
	ufloat_g_t Ly0g __attribute__((unused)) = Ly/(ufloat_g_t)G_BIN_DENSITY;
	ufloat_g_t Lz0g __attribute__((unused)) = Lz/(ufloat_g_t)G_BIN_DENSITY;
	ufloat_g_t eps __attribute__((unused)) = 1e-5;                            // An epsilon for the 3D binning.
	if (std::is_same<ufloat_g_t, float>::value) eps = FLT_EPSILON;
	if (std::is_same<ufloat_g_t, double>::value) eps = DBL_EPSILON;
	bool C2D = false; if (G_BIN_APPROACH==0) C2D = true;
	bool C3D = false; if (G_BIN_APPROACH==0) C3D = true;
	
	// Proceed only if there are actual faces loaded in the current object.
	tic_simple("");
	if (v_geom_f_face_1_X.size() > 0)
	{
		// Declare and allocate std::vector<int> bin arrays, which will be updated during traversal.
		int n_bins_v = 1; for (int d = 0; d < N_DIM-1; d++) n_bins_v *= G_BIN_DENSITY;
		int n_bins_b = 1; for (int d = 0; d < N_DIM; d++)   n_bins_b *= G_BIN_DENSITY;
		std::vector<int> *bins_a_2D = new std::vector<int>[n_bins_v];
		std::vector<int> *bins_a_3D = new std::vector<int>[n_bins_b];
		
		// Traverse faces and identify the bins they should go in.
		std::cout << "Starting CPU binning..." << std::endl;
		cudaDeviceSynchronize();
		for (int j = 0; j < n_faces_a[i_dev]; j++)
		{
			if (j < n_faces[i_dev])
			{
				// Load face vertices from coordinate list.
				ufloat_g_t vx1 = geom_f_face_X[i_dev][j + 0*n_faces_a[i_dev]];
				ufloat_g_t vy1 = geom_f_face_X[i_dev][j + 1*n_faces_a[i_dev]];
				ufloat_g_t vz1 = geom_f_face_X[i_dev][j + 2*n_faces_a[i_dev]];
				ufloat_g_t vx2 = geom_f_face_X[i_dev][j + 3*n_faces_a[i_dev]];
				ufloat_g_t vy2 = geom_f_face_X[i_dev][j + 4*n_faces_a[i_dev]];
				ufloat_g_t vz2 = geom_f_face_X[i_dev][j + 5*n_faces_a[i_dev]];
				ufloat_g_t vx3 = geom_f_face_X[i_dev][j + 6*n_faces_a[i_dev]];
				ufloat_g_t vy3 = geom_f_face_X[i_dev][j + 7*n_faces_a[i_dev]];
				ufloat_g_t vz3 = geom_f_face_X[i_dev][j + 8*n_faces_a[i_dev]];
				
				if (N_DIM==2)
				{
					// Get the bounding box.
					bool C = true;
					ufloat_g_t vBx_m = std::min({vx1,vx2});
					ufloat_g_t vBx_M = std::max({vx1,vx2});
					ufloat_g_t vBy_m = std::min({vy1,vy2});
					ufloat_g_t vBy_M = std::max({vy1,vy2});
					if (vBx_m<-dx&&vBx_M<-dx || vBx_m>Lx+dx&&vBx_M>Lx+dx)
						C = false;
					if (vBy_m<-dx&&vBy_M<-dx || vBy_m>Ly+dx&&vBy_M>Ly+dx)
						C = false;
					
					// Identify the bin indices of the lower and upper bounds.
					int bin_id_xl = std::max((int)(vBx_m*G_BIN_DENSITY)-1, 0);
					int bin_id_yl = std::max((int)(vBy_m*G_BIN_DENSITY)-1, 0);
					int bin_id_xL = std::min((int)(vBx_M*G_BIN_DENSITY)+2, G_BIN_DENSITY);
					int bin_id_yL = std::min((int)(vBy_M*G_BIN_DENSITY)+2, G_BIN_DENSITY);
					
					// Traverse bin indices and add this face to the corresponding vectors.
					if (C)
					{
						for (int J = bin_id_yl; J < bin_id_yL; J++)
						{
							for (int I = bin_id_xl; I < bin_id_xL; I++)
							{
								if (G_BIN_APPROACH==1)
									C3D = IncludeInBin<ufloat_g_t,2>(I*Lx0g-dx,(I+1)*Lx0g+dx,J*Ly0g-dx,(J+1)*Ly0g+dx,0,0,vBx_m,vBx_M,vBy_m,vBy_M,0,0,vx1,vy1,0,vx2,vy2,0,0,0,0);
								
								if (C3D)
									bins_a_3D[I+G_BIN_DENSITY*J].push_back(j);
							}
							
							if (G_BIN_APPROACH==1)
								C2D = IncludeInBin<ufloat_g_t,2>(-eps,Lx+eps,J*Ly0g-eps,(J+1)*Ly0g+eps,0,0,vBx_m,vBx_M,vBy_m,vBy_M,0,0,vx1,vy1,0,vx2,vy2,0,0,0,0);
							
							if (C2D)
								bins_a_2D[J].push_back(j);
						}
					}
				}
				else // N_DIM==3
				{
					// Get bounding box (safe version)
					bool C = true;
					ufloat_g_t vBx_m = std::min(std::min(vx1, vx2), vx3);
					ufloat_g_t vBx_M = std::max(std::max(vx1, vx2), vx3);
					ufloat_g_t vBy_m = std::min(std::min(vy1, vy2), vy3);
					ufloat_g_t vBy_M = std::max(std::max(vy1, vy2), vy3);
					ufloat_g_t vBz_m = std::min(std::min(vz1, vz2), vz3);
					ufloat_g_t vBz_M = std::max(std::max(vz1, vz2), vz3);
					if (vBx_m<-dx&&vBx_M<-dx || vBx_m>Lx+dx&&vBx_M>Lx+dx)
						C = false;
					if (vBy_m<-dx&&vBy_M<-dx || vBy_m>Ly+dx&&vBy_M>Ly+dx)
						C = false;
					if (vBz_m<-dx&&vBz_M<-dx || vBz_m>Lz+dx&&vBz_M>Lz+dx)
						C = false;
					
					// Identify the bin indices of the lower and upper bounds.
					int bin_id_xl = std::max((int)(vBx_m*G_BIN_DENSITY)-1, 0);
					int bin_id_yl = std::max((int)(vBy_m*G_BIN_DENSITY)-1, 0);
					int bin_id_zl = std::max((int)(vBz_m*G_BIN_DENSITY)-1, 0);
					int bin_id_xL = std::min((int)(vBx_M*G_BIN_DENSITY)+2, G_BIN_DENSITY);
					int bin_id_yL = std::min((int)(vBy_M*G_BIN_DENSITY)+2, G_BIN_DENSITY);
					int bin_id_zL = std::min((int)(vBz_M*G_BIN_DENSITY)+2, G_BIN_DENSITY);
					
					// Traverse bin indices and add this face to the corresponding vectors.
					if (C)
					{
						for (int K = bin_id_zl; K < bin_id_zL; K++)
						{
							for (int J = bin_id_yl; J < bin_id_yL; J++)
							{
								for (int I = bin_id_xl; I < bin_id_xL; I++)
								{
									if (G_BIN_APPROACH==1)
										C3D = IncludeInBin<ufloat_g_t,3>(I*Lx0g-dx,(I+1)*Lx0g+dx,J*Ly0g-dx,(J+1)*Ly0g+dx,K*Lz0g-dx,(K+1)*Lz0g+dx,vBx_m,vBx_M,vBy_m,vBy_M,vBz_m,vBz_M,vx1,vy1,vz1,vx2,vy2,vz2,vx3,vy3,vz3);
									
									if (C3D)
										bins_a_3D[I+G_BIN_DENSITY*J+G_BIN_DENSITY*G_BIN_DENSITY*K].push_back(j);
								}
								
								if (G_BIN_APPROACH==1)
									C2D = IncludeInBin<ufloat_g_t,3>(-eps,Lx+eps,J*Ly0g-eps,(J+1)*Ly0g+eps,K*Lz0g-eps,(K+1)*Lz0g+eps,vBx_m,vBx_M,vBy_m,vBy_M,vBz_m,vBz_M,vx1,vy1,vz1,vx2,vy2,vz2,vx3,vy3,vz3);
									
								if (C2D)
									bins_a_2D[J+G_BIN_DENSITY*K].push_back(j);
							}
						}
					}
				}
			}
			
		}
		cudaDeviceSynchronize();
		std::cout << "Elapsed time (CPU): " << toc_simple("",T_US) << std::endl;
		
		
		// Insert binned faces into GPU memory.
		std::vector<int> bins_n_v;
		std::vector<int> bins_N_v;
		std::vector<int> bins_f_v;
		std::vector<int> bins_n_b;
		std::vector<int> bins_N_b;
		std::vector<int> bins_f_b;
		int Npv = 0;
		int Npb = 0;
		const int PADDING = 4;
		for (int p = 0; p < n_bins_v; p++)
		{
			int npv = bins_a_2D[p].size();
			bins_n_v.push_back(npv);
			bins_N_v.push_back(Npv);
			if (npv > 0)
			{
				for (int k = 0; k < npv; k++)
					bins_f_v.push_back(bins_a_2D[p][k]);
				
				int rem = PADDING-npv%PADDING;
				for (int k = 0; k < rem; k++)
					bins_f_v.push_back(-1);
				
				Npv += npv + rem;
			}
		}
		for (int p = 0; p < n_bins_b; p++)
		{
			int npb = bins_a_3D[p].size();
			bins_n_b.push_back(npb);
			bins_N_b.push_back(Npb);
			if (npb > 0)
			{
				for (int k = 0; k < npb; k++)
					bins_f_b.push_back(bins_a_3D[p][k]);
				
				int rem = PADDING-npb%PADDING;
				for (int k = 0; k < rem; k++)
					bins_f_b.push_back(-1);
				
				Npb += npb + rem;
			}
		}
		
		// Now copy final vector data to the GPU.
		// 2D
		binned_face_ids_n_v[i_dev] = new int[n_bins_v];
		binned_face_ids_N_v[i_dev] = new int[n_bins_v];
		binned_face_ids_v[i_dev] = new int[bins_f_v.size()];
		gpuErrchk( cudaMalloc((void **)&c_binned_face_ids_n_v[i_dev], n_bins_v*sizeof(int)) );
		gpuErrchk( cudaMalloc((void **)&c_binned_face_ids_N_v[i_dev], n_bins_v*sizeof(int)) );
		gpuErrchk( cudaMalloc((void **)&c_binned_face_ids_v[i_dev], bins_f_v.size()*sizeof(int)) );
		for (int p = 0; p < n_bins_v; p++)
		{
			binned_face_ids_n_v[i_dev][p] = bins_n_v[p];
			binned_face_ids_N_v[i_dev][p] = bins_N_v[p];
		}
		for (int p = 0; p < bins_f_v.size(); p++)
		{
			binned_face_ids_v[i_dev][p] = bins_f_v[p];
		}
		gpuErrchk( cudaMemcpy(c_binned_face_ids_n_v[i_dev], binned_face_ids_n_v[i_dev], n_bins_v*sizeof(int), cudaMemcpyHostToDevice) );
		gpuErrchk( cudaMemcpy(c_binned_face_ids_N_v[i_dev], binned_face_ids_N_v[i_dev], n_bins_v*sizeof(int), cudaMemcpyHostToDevice) );
		gpuErrchk( cudaMemcpy(c_binned_face_ids_v[i_dev], binned_face_ids_v[i_dev], bins_f_v.size()*sizeof(int), cudaMemcpyHostToDevice) );
		// 3D
		binned_face_ids_n_b[i_dev] = new int[n_bins_b];
		binned_face_ids_N_b[i_dev] = new int[n_bins_b];
		binned_face_ids_b[i_dev] = new int[bins_f_b.size()];
		gpuErrchk( cudaMalloc((void **)&c_binned_face_ids_n_b[i_dev], n_bins_b*sizeof(int)) );
		gpuErrchk( cudaMalloc((void **)&c_binned_face_ids_N_b[i_dev], n_bins_b*sizeof(int)) );
		gpuErrchk( cudaMalloc((void **)&c_binned_face_ids_b[i_dev], bins_f_b.size()*sizeof(int)) );
		for (int p = 0; p < n_bins_b; p++)
		{
			binned_face_ids_n_b[i_dev][p] = bins_n_b[p];
			binned_face_ids_N_b[i_dev][p] = bins_N_b[p];
		}
		for (int p = 0; p < bins_f_b.size(); p++)
		{
			binned_face_ids_b[i_dev][p] = bins_f_b[p];
		}
		gpuErrchk( cudaMemcpy(c_binned_face_ids_n_b[i_dev], binned_face_ids_n_b[i_dev], n_bins_b*sizeof(int), cudaMemcpyHostToDevice) );
		gpuErrchk( cudaMemcpy(c_binned_face_ids_N_b[i_dev], binned_face_ids_N_b[i_dev], n_bins_b*sizeof(int), cudaMemcpyHostToDevice) );
		gpuErrchk( cudaMemcpy(c_binned_face_ids_b[i_dev], binned_face_ids_b[i_dev], bins_f_b.size()*sizeof(int), cudaMemcpyHostToDevice) );
		
		
		// DEBUG (2D)
// 		std::cout << "Finished CPU binning, starting debugging..." << std::endl;
// 		std::cout << "APPROACH: ALT 2D" << std::endl;
// 		for (int p = 0; p < n_bins_v; p++)
// 		{
// 			int Nbpv = binned_face_ids_N_v[i_dev][p];
// 			int npbv = binned_face_ids_n_v[i_dev][p];
// 			int npb = bins_a_2D[p].size();
// 			if (npb > 0)
// 			{
// 				std::cout << "Bin #" << p << ": ";
// 				bool same = true;
// 				
// 				if (npb != npbv)
// 					same = false;
// 				else
// 				{
// 					for (int K = 0; K < npb; K++)
// 					{
// 						if (bins_a_2D[p][K] != binned_face_ids_v[i_dev][Nbpv + K])
// 							same = false;
// 					}
// 				}
// 				if (same)
// 					std::cout << "SAME" << std::endl;
// 				else
// 					std::cout << "NOT THE SAME (" << npb-npbv << ")" << std::endl;
// 			}
// 		}
		// DEBUG (3D)
// 		std::cout << "APPROACH: ALT 3D" << std::endl;
// 		for (int p = 0; p < n_bins_b; p++)
// 		{
// 			int Nbpv = binned_face_ids_N_b[i_dev][p];
// 			int npbv = binned_face_ids_n_b[i_dev][p];
// 			int npb = bins_a_3D[p].size();
// 			if (npb > 0)
// 			{
// 				std::cout << "Bin #" << p << ": ";
// 				bool same = true;
// 				
// 				if (npb != npbv)
// 					same = false;
// 				else
// 				{
// 					for (int K = 0; K < npb; K++)
// 					{
// 						if (bins_a_3D[p][K] != binned_face_ids_b[i_dev][Nbpv + K])
// 							same = false;
// 					}
// 				}
// 				if (same)
// 					std::cout << "SAME" << std::endl;
// 				else
// 					std::cout << "NOT THE SAME (" << npb-npbv << ")" << std::endl;
// 			}
// 		}
		
		
		// Free memory used for CPU-side bin arrays.
		delete[] bins_a_2D;
		delete[] bins_a_3D;
	}
	else
	{
		std::cout << "ERROR: Could not make bins...there are no faces..." << std::endl;
	}
	
	return 0;
}


