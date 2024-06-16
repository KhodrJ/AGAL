// Author: Khodr Jaber
// First Completed: May 20, 2024
// Last Updated: May 20, 2024



/*
888    888                        888                  
888    888                        888                  
888    888                        888                  
8888888888  .d88b.   8888b.   .d88888  .d88b.  888d888 
888    888 d8P  Y8b     "88b d88" 888 d8P  Y8b 888P"   
888    888 88888888 .d888888 888  888 88888888 888     
888    888 Y8b.     888  888 Y88b 888 Y8b.     888     
888    888  "Y8888  "Y888888  "Y88888  "Y8888  888     
*/

#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>

// Input variables.
#ifndef Nx
	#define Nx 128
#endif
#ifndef N_Q
	#define N_Q 9
	#define N_DIM 2
	#define N_Q_max 9
#endif
#ifndef N_PRECISION
	#define N_PRECISION 0
#endif
#ifndef N_iters
	#define N_iters (1*Nx)
#endif
#ifndef N_CONN_TYPE
	#define N_CONN_TYPE 0
#endif
#ifndef N_OUTPUT_TYPE
	#define N_OUTPUT_TYPE 1
#endif
#ifndef N_SHUFFLE
	#define N_SHUFFLE 0
#endif
#ifndef N_OCTREE
	#define N_OCTREE 0
#endif

// Other variables.
#define v0 5e-5
#define M_BLOCK 128
#define T_S 0
#define T_MS 1
#define T_US 2

// For variable floating-point precision at compile time.
#define APPEND(x, y) x ## y
#if (N_PRECISION==1)
	typedef double ufloat_t;
	#define N_Pf(x) (x)
#else
	typedef float ufloat_t;
	#define N_Pf(x) (APPEND(x,F))
#endif

// Used for debugging CUDA kernels.
// Credit: https://stackoverflow.com/a/14038590 (username: talonmies)
#define gpuErrchk(ans) {gpuAssert((ans), __FILE__, __LINE__);}
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
        if (code != cudaSuccess)
        {
                fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
                if (abort)
                        exit(code);
        }
}

// Another version of tic-toc for separate measurements, controls whether to output to terminal.
// Credit: https://stackoverflow.com/a/63391159 (username: mma)
inline double tic_simple(std::string s="", int mode=0, int scale=0, int print=1)
{
	static std::chrono::_V2::system_clock::time_point t_start_simple;
    
	if (mode==0)
	{
		t_start_simple = std::chrono::high_resolution_clock::now();
		return 0.0;
	}
	else
	{
		auto t_end_simple = std::chrono::high_resolution_clock::now();
		if (scale == T_S)
		{
			if (print)
				std::cout << s << ": " << std::chrono::duration_cast<std::chrono::seconds>(t_end_simple-t_start_simple).count() << " seconds\n";
			return std::chrono::duration_cast<std::chrono::seconds>(t_end_simple-t_start_simple).count();
		}
		else if (scale == T_MS)
		{
			if (print)
				std::cout << s << ": " << std::chrono::duration_cast<std::chrono::milliseconds>(t_end_simple-t_start_simple).count() << " milliseconds\n";
			return std::chrono::duration_cast<std::chrono::milliseconds>(t_end_simple-t_start_simple).count();
		}
		else
		{
			if (print)
				std::cout << s << ": " << std::chrono::duration_cast<std::chrono::microseconds>(t_end_simple-t_start_simple).count() << " microseconds\n";
			return std::chrono::duration_cast<std::chrono::microseconds>(t_end_simple-t_start_simple).count();
		}
	}
}
inline double toc_simple(std::string s, int scale=T_S, int print=1) { return tic_simple(s,1,scale,print); }



/*
 .d8888b.           888                           
d88P  Y88b          888                           
Y88b.               888                           
 "Y888b.    .d88b.  888 888  888  .d88b.  888d888 
    "Y88b. d88""88b 888 888  888 d8P  Y8b 888P"   
      "888 888  888 888 Y88  88P 88888888 888     
Y88b  d88P Y88..88P 888  Y8bd8P  Y8b.     888     
 "Y8888P"   "Y88P"  888   Y88P    "Y8888  888     
*/



__global__
void Cu_SetInitialConditions
(
	int n_maxcells,
	ufloat_t *cells_f_F,
	ufloat_t rho_t0, ufloat_t u_t0, ufloat_t v_t0, ufloat_t w_t0
)
{
	int kap = blockIdx.x*blockDim.x + threadIdx.x;

	if (kap < n_maxcells)
	{
		// Set initial conditions and write.
#if (N_Q==9)
		ufloat_t cdotu = N_Pf(0.0);
		ufloat_t udotu = u_t0*u_t0 + v_t0*v_t0;
		cdotu = 0; cells_f_F[kap + 0*n_maxcells] = N_Pf(0.444444444444444)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
		cdotu = +u_t0; cells_f_F[kap + 3*n_maxcells] = N_Pf(0.111111111111111)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
		cdotu = +v_t0; cells_f_F[kap + 4*n_maxcells] = N_Pf(0.111111111111111)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
		cdotu = -u_t0; cells_f_F[kap + 1*n_maxcells] = N_Pf(0.111111111111111)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
		cdotu = -v_t0; cells_f_F[kap + 2*n_maxcells] = N_Pf(0.111111111111111)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
		cdotu = +u_t0+v_t0; cells_f_F[kap + 7*n_maxcells] = N_Pf(0.027777777777778)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
		cdotu = -u_t0+v_t0; cells_f_F[kap + 8*n_maxcells] = N_Pf(0.027777777777778)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
		cdotu = -u_t0-v_t0; cells_f_F[kap + 5*n_maxcells] = N_Pf(0.027777777777778)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
		cdotu = +u_t0-v_t0; cells_f_F[kap + 6*n_maxcells] = N_Pf(0.027777777777778)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
#elif (N_Q==19)
		ufloat_t cdotu = N_Pf(0.0);
		ufloat_t udotu = u_t0*u_t0 + v_t0*v_t0 + w_t0*w_t0;
		cdotu = 0; cells_f_F[kap + 0*n_maxcells] = N_Pf(0.333333333333333)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
		cdotu = +u_t0; cells_f_F[kap + 2*n_maxcells] = N_Pf(0.055555555555556)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
		cdotu = -u_t0; cells_f_F[kap + 1*n_maxcells] = N_Pf(0.055555555555556)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
		cdotu = +v_t0; cells_f_F[kap + 4*n_maxcells] = N_Pf(0.055555555555556)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
		cdotu = -v_t0; cells_f_F[kap + 3*n_maxcells] = N_Pf(0.055555555555556)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
		cdotu = +w_t0; cells_f_F[kap + 6*n_maxcells] = N_Pf(0.055555555555556)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
		cdotu = -w_t0; cells_f_F[kap + 5*n_maxcells] = N_Pf(0.055555555555556)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
		cdotu = +u_t0+v_t0; cells_f_F[kap + 8*n_maxcells] = N_Pf(0.027777777777778)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
		cdotu = -u_t0-v_t0; cells_f_F[kap + 7*n_maxcells] = N_Pf(0.027777777777778)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
		cdotu = +u_t0+w_t0; cells_f_F[kap + 10*n_maxcells] = N_Pf(0.027777777777778)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
		cdotu = -u_t0-w_t0; cells_f_F[kap + 9*n_maxcells] = N_Pf(0.027777777777778)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
		cdotu = +v_t0+w_t0; cells_f_F[kap + 12*n_maxcells] = N_Pf(0.027777777777778)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
		cdotu = -v_t0-w_t0; cells_f_F[kap + 11*n_maxcells] = N_Pf(0.027777777777778)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
		cdotu = +u_t0-v_t0; cells_f_F[kap + 14*n_maxcells] = N_Pf(0.027777777777778)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
		cdotu = -u_t0+v_t0; cells_f_F[kap + 13*n_maxcells] = N_Pf(0.027777777777778)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
		cdotu = +u_t0-w_t0; cells_f_F[kap + 16*n_maxcells] = N_Pf(0.027777777777778)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
		cdotu = -u_t0+w_t0; cells_f_F[kap + 15*n_maxcells] = N_Pf(0.027777777777778)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
		cdotu = +v_t0-w_t0; cells_f_F[kap + 18*n_maxcells] = N_Pf(0.027777777777778)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
		cdotu = -v_t0+w_t0; cells_f_F[kap + 17*n_maxcells] = N_Pf(0.027777777777778)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
#else // N_Q==27
		ufloat_t cdotu = N_Pf(0.0);
		ufloat_t udotu = u_t0*u_t0 + v_t0*v_t0 + w_t0*w_t0;
		cdotu = 0; cells_f_F[kap + 0*n_maxcells] = N_Pf(0.296296296296296)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
		cdotu = +u_t0; cells_f_F[kap + 2*n_maxcells] = N_Pf(0.074074074074074)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
		cdotu = -u_t0; cells_f_F[kap + 1*n_maxcells] = N_Pf(0.074074074074074)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
		cdotu = +v_t0; cells_f_F[kap + 4*n_maxcells] = N_Pf(0.074074074074074)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
		cdotu = -v_t0; cells_f_F[kap + 3*n_maxcells] = N_Pf(0.074074074074074)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
		cdotu = +w_t0; cells_f_F[kap + 6*n_maxcells] = N_Pf(0.074074074074074)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
		cdotu = -w_t0; cells_f_F[kap + 5*n_maxcells] = N_Pf(0.074074074074074)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
		cdotu = +u_t0+v_t0; cells_f_F[kap + 8*n_maxcells] = N_Pf(0.018518518518519)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
		cdotu = -u_t0-v_t0; cells_f_F[kap + 7*n_maxcells] = N_Pf(0.018518518518519)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
		cdotu = +u_t0+w_t0; cells_f_F[kap + 10*n_maxcells] = N_Pf(0.018518518518519)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
		cdotu = -u_t0-w_t0; cells_f_F[kap + 9*n_maxcells] = N_Pf(0.018518518518519)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
		cdotu = +v_t0+w_t0; cells_f_F[kap + 12*n_maxcells] = N_Pf(0.018518518518519)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
		cdotu = -v_t0-w_t0; cells_f_F[kap + 11*n_maxcells] = N_Pf(0.018518518518519)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
		cdotu = +u_t0-v_t0; cells_f_F[kap + 14*n_maxcells] = N_Pf(0.018518518518519)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
		cdotu = -u_t0+v_t0; cells_f_F[kap + 13*n_maxcells] = N_Pf(0.018518518518519)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
		cdotu = +u_t0-w_t0; cells_f_F[kap + 16*n_maxcells] = N_Pf(0.018518518518519)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
		cdotu = -u_t0+w_t0; cells_f_F[kap + 15*n_maxcells] = N_Pf(0.018518518518519)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
		cdotu = +v_t0-w_t0; cells_f_F[kap + 18*n_maxcells] = N_Pf(0.018518518518519)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
		cdotu = -v_t0+w_t0; cells_f_F[kap + 17*n_maxcells] = N_Pf(0.018518518518519)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
		cdotu = +u_t0+v_t0+w_t0; cells_f_F[kap + 20*n_maxcells] = N_Pf(0.004629629629630)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
		cdotu = -u_t0-v_t0-w_t0; cells_f_F[kap + 19*n_maxcells] = N_Pf(0.004629629629630)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
		cdotu = +u_t0+v_t0-w_t0; cells_f_F[kap + 22*n_maxcells] = N_Pf(0.004629629629630)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
		cdotu = -u_t0-v_t0+w_t0; cells_f_F[kap + 21*n_maxcells] = N_Pf(0.004629629629630)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
		cdotu = +u_t0-v_t0+w_t0; cells_f_F[kap + 24*n_maxcells] = N_Pf(0.004629629629630)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
		cdotu = -u_t0+v_t0-w_t0; cells_f_F[kap + 23*n_maxcells] = N_Pf(0.004629629629630)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
		cdotu = -u_t0+v_t0+w_t0; cells_f_F[kap + 26*n_maxcells] = N_Pf(0.004629629629630)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
		cdotu = +u_t0-v_t0-w_t0; cells_f_F[kap + 25*n_maxcells] = N_Pf(0.004629629629630)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu);
#endif
	}
}

__global__
void Cu_Collide
(
	int n_maxcells, ufloat_t otau,
	int *cells_ID_nbr, ufloat_t *cells_f_F
)
{
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	//int I_kap = kap % Nx;
	int J_kap = (kap / Nx) % Nx;
#if (N_DIM==3)
	int K_kap = (kap / Nx) / Nx;
#endif
	
	if (kap < n_maxcells)
	{
#if (N_Q==9)
		// Load DDFs, compute macroscopic properties.
		ufloat_t f_0 = cells_f_F[kap + 0*n_maxcells];
		ufloat_t f_1 = cells_f_F[kap + 3*n_maxcells];
		ufloat_t f_2 = cells_f_F[kap + 4*n_maxcells];
		ufloat_t f_3 = cells_f_F[kap + 1*n_maxcells];
		ufloat_t f_4 = cells_f_F[kap + 2*n_maxcells];
		ufloat_t f_5 = cells_f_F[kap + 7*n_maxcells];
		ufloat_t f_6 = cells_f_F[kap + 8*n_maxcells];
		ufloat_t f_7 = cells_f_F[kap + 5*n_maxcells];
		ufloat_t f_8 = cells_f_F[kap + 6*n_maxcells];
		ufloat_t rho_kap = +f_0 +f_1 +f_2 +f_3 +f_4 +f_5 +f_6 +f_7 +f_8;
		ufloat_t u_kap = ( +f_1 -f_3 +f_5 -f_6 -f_7 +f_8) / rho_kap;
		ufloat_t v_kap = ( +f_2 -f_4 +f_5 +f_6 -f_7 -f_8) / rho_kap;
		ufloat_t cdotu = N_Pf(0.0);
		ufloat_t udotu = u_kap*u_kap + v_kap*v_kap;
		ufloat_t omeg = otau;
		ufloat_t omegp = N_Pf(1.0) - omeg;
		
		// Collision step.
		cdotu = N_Pf(0.0); f_0 = f_0*omegp + ( N_Pf(0.444444444444444)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		cdotu = +u_kap; f_1 = f_1*omegp + ( N_Pf(0.111111111111111)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		cdotu = +v_kap; f_2 = f_2*omegp + ( N_Pf(0.111111111111111)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		cdotu = -u_kap; f_3 = f_3*omegp + ( N_Pf(0.111111111111111)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		cdotu = -v_kap; f_4 = f_4*omegp + ( N_Pf(0.111111111111111)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		cdotu = +u_kap+v_kap; f_5 = f_5*omegp + ( N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		cdotu = -u_kap+v_kap; f_6 = f_6*omegp + ( N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		cdotu = -u_kap-v_kap; f_7 = f_7*omegp + ( N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		cdotu = +u_kap-v_kap; f_8 = f_8*omegp + ( N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		
		// Impose BC (only checking +y in 2D for the lid).
#if (N_CONN_TYPE==0)
		if (J_kap == Nx-1)
#else
		if (cells_ID_nbr[kap + 2*n_maxcells] < 0)
#endif
		{
			// p = 2
			cdotu = N_Pf(0.000000000000000); f_2 = f_2 - N_Pf(2.0)*N_Pf(0.111111111111111)*N_Pf(3.0)*cdotu;
			
			// p = 5
			cdotu = N_Pf(0.050000000000000); f_5 = f_5 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
			
			// p = 6
			cdotu = N_Pf(-0.050000000000000); f_6 = f_6 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
		}
		
		// Store DDFs.
		cells_f_F[kap + 0*n_maxcells] = f_0;
		cells_f_F[kap + 1*n_maxcells] = f_1;
		cells_f_F[kap + 2*n_maxcells] = f_2;
		cells_f_F[kap + 3*n_maxcells] = f_3;
		cells_f_F[kap + 4*n_maxcells] = f_4;
		cells_f_F[kap + 5*n_maxcells] = f_5;
		cells_f_F[kap + 6*n_maxcells] = f_6;
		cells_f_F[kap + 7*n_maxcells] = f_7;
		cells_f_F[kap + 8*n_maxcells] = f_8;
		
#elif (N_Q==19)
		// Load DDFs, compute macroscopic properties.
		ufloat_t f_0 = cells_f_F[kap + 0*n_maxcells];
		ufloat_t f_1 = cells_f_F[kap + 2*n_maxcells];
		ufloat_t f_2 = cells_f_F[kap + 1*n_maxcells];
		ufloat_t f_3 = cells_f_F[kap + 4*n_maxcells];
		ufloat_t f_4 = cells_f_F[kap + 3*n_maxcells];
		ufloat_t f_5 = cells_f_F[kap + 6*n_maxcells];
		ufloat_t f_6 = cells_f_F[kap + 5*n_maxcells];
		ufloat_t f_7 = cells_f_F[kap + 8*n_maxcells];
		ufloat_t f_8 = cells_f_F[kap + 7*n_maxcells];
		ufloat_t f_9 = cells_f_F[kap + 10*n_maxcells];
		ufloat_t f_10 = cells_f_F[kap + 9*n_maxcells];
		ufloat_t f_11 = cells_f_F[kap + 12*n_maxcells];
		ufloat_t f_12 = cells_f_F[kap + 11*n_maxcells];
		ufloat_t f_13 = cells_f_F[kap + 14*n_maxcells];
		ufloat_t f_14 = cells_f_F[kap + 13*n_maxcells];
		ufloat_t f_15 = cells_f_F[kap + 16*n_maxcells];
		ufloat_t f_16 = cells_f_F[kap + 15*n_maxcells];
		ufloat_t f_17 = cells_f_F[kap + 18*n_maxcells];
		ufloat_t f_18 = cells_f_F[kap + 17*n_maxcells];
		ufloat_t rho_kap = +f_0 +f_1 +f_2 +f_3 +f_4 +f_5 +f_6 +f_7 +f_8 +f_9 +f_10 +f_11 +f_12 +f_13 +f_14 +f_15 +f_16 +f_17 +f_18;
		ufloat_t u_kap = ( +f_1 -f_2 +f_7 -f_8 +f_9 -f_10 +f_13 -f_14 +f_15 -f_16) / rho_kap;
		ufloat_t v_kap = ( +f_3 -f_4 +f_7 -f_8 +f_11 -f_12 -f_13 +f_14 +f_17 -f_18) / rho_kap;
		ufloat_t w_kap = ( +f_5 -f_6 +f_9 -f_10 +f_11 -f_12 -f_15 +f_16 -f_17 +f_18) / rho_kap;
		ufloat_t cdotu = N_Pf(0.0);
		ufloat_t udotu = u_kap*u_kap + v_kap*v_kap + w_kap*w_kap;
		ufloat_t omeg = otau;
		ufloat_t omegp = N_Pf(1.0) - omeg;
		
		// Collision step.
		cdotu = N_Pf(0.0); f_0 = f_0*omegp + ( N_Pf(0.333333333333333)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		cdotu = +u_kap; f_1 = f_1*omegp + ( N_Pf(0.055555555555556)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		cdotu = -u_kap; f_2 = f_2*omegp + ( N_Pf(0.055555555555556)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		cdotu = +v_kap; f_3 = f_3*omegp + ( N_Pf(0.055555555555556)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		cdotu = -v_kap; f_4 = f_4*omegp + ( N_Pf(0.055555555555556)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		cdotu = +w_kap; f_5 = f_5*omegp + ( N_Pf(0.055555555555556)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		cdotu = -w_kap; f_6 = f_6*omegp + ( N_Pf(0.055555555555556)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		cdotu = +u_kap+v_kap; f_7 = f_7*omegp + ( N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		cdotu = -u_kap-v_kap; f_8 = f_8*omegp + ( N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		cdotu = +u_kap+w_kap; f_9 = f_9*omegp + ( N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		cdotu = -u_kap-w_kap; f_10 = f_10*omegp + ( N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		cdotu = +v_kap+w_kap; f_11 = f_11*omegp + ( N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		cdotu = -v_kap-w_kap; f_12 = f_12*omegp + ( N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		cdotu = +u_kap-v_kap; f_13 = f_13*omegp + ( N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		cdotu = -u_kap+v_kap; f_14 = f_14*omegp + ( N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		cdotu = +u_kap-w_kap; f_15 = f_15*omegp + ( N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		cdotu = -u_kap+w_kap; f_16 = f_16*omegp + ( N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		cdotu = +v_kap-w_kap; f_17 = f_17*omegp + ( N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		cdotu = -v_kap+w_kap; f_18 = f_18*omegp + ( N_Pf(0.027777777777778)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		
		// Impose BC (only checking +y in 2D for the lid).
#if (N_CONN_TYPE==0)
		if (K_kap == Nx-1)
#else
		if (cells_ID_nbr[kap + 5*n_maxcells] < 0)
#endif
		{
			// p = 5.
			cdotu = N_Pf(0.000000000000000); f_5 = f_5 - N_Pf(2.0)*N_Pf(0.055555555555556)*N_Pf(3.0)*cdotu;
			
			// p = 9.
			cdotu = N_Pf(0.050000000000000); f_9 = f_9 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
			
			// p = 11.
			cdotu = N_Pf(0.000000000000000); f_11 = f_11 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
			
			// p = 16.
			cdotu = N_Pf(-0.050000000000000); f_16 = f_16 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
			
			// p = 18.
			cdotu = N_Pf(0.000000000000000); f_18 = f_18 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
		
		}
		
		// Store DDFs.
		cells_f_F[kap + 0*n_maxcells] = f_0;
		cells_f_F[kap + 1*n_maxcells] = f_1;
		cells_f_F[kap + 2*n_maxcells] = f_2;
		cells_f_F[kap + 3*n_maxcells] = f_3;
		cells_f_F[kap + 4*n_maxcells] = f_4;
		cells_f_F[kap + 5*n_maxcells] = f_5;
		cells_f_F[kap + 6*n_maxcells] = f_6;
		cells_f_F[kap + 7*n_maxcells] = f_7;
		cells_f_F[kap + 8*n_maxcells] = f_8;
		cells_f_F[kap + 9*n_maxcells] = f_9;
		cells_f_F[kap + 10*n_maxcells] = f_10;
		cells_f_F[kap + 11*n_maxcells] = f_11;
		cells_f_F[kap + 12*n_maxcells] = f_12;
		cells_f_F[kap + 13*n_maxcells] = f_13;
		cells_f_F[kap + 14*n_maxcells] = f_14;
		cells_f_F[kap + 15*n_maxcells] = f_15;
		cells_f_F[kap + 16*n_maxcells] = f_16;
		cells_f_F[kap + 17*n_maxcells] = f_17;
		cells_f_F[kap + 18*n_maxcells] = f_18;
#else // N_Q==27
		// Load DDFs, compute macroscopic properties.
		ufloat_t f_0 = cells_f_F[kap + 0*n_maxcells];
		ufloat_t f_1 = cells_f_F[kap + 2*n_maxcells];
		ufloat_t f_2 = cells_f_F[kap + 1*n_maxcells];
		ufloat_t f_3 = cells_f_F[kap + 4*n_maxcells];
		ufloat_t f_4 = cells_f_F[kap + 3*n_maxcells];
		ufloat_t f_5 = cells_f_F[kap + 6*n_maxcells];
		ufloat_t f_6 = cells_f_F[kap + 5*n_maxcells];
		ufloat_t f_7 = cells_f_F[kap + 8*n_maxcells];
		ufloat_t f_8 = cells_f_F[kap + 7*n_maxcells];
		ufloat_t f_9 = cells_f_F[kap + 10*n_maxcells];
		ufloat_t f_10 = cells_f_F[kap + 9*n_maxcells];
		ufloat_t f_11 = cells_f_F[kap + 12*n_maxcells];
		ufloat_t f_12 = cells_f_F[kap + 11*n_maxcells];
		ufloat_t f_13 = cells_f_F[kap + 14*n_maxcells];
		ufloat_t f_14 = cells_f_F[kap + 13*n_maxcells];
		ufloat_t f_15 = cells_f_F[kap + 16*n_maxcells];
		ufloat_t f_16 = cells_f_F[kap + 15*n_maxcells];
		ufloat_t f_17 = cells_f_F[kap + 18*n_maxcells];
		ufloat_t f_18 = cells_f_F[kap + 17*n_maxcells];
		ufloat_t f_19 = cells_f_F[kap + 20*n_maxcells];
		ufloat_t f_20 = cells_f_F[kap + 19*n_maxcells];
		ufloat_t f_21 = cells_f_F[kap + 22*n_maxcells];
		ufloat_t f_22 = cells_f_F[kap + 21*n_maxcells];
		ufloat_t f_23 = cells_f_F[kap + 24*n_maxcells];
		ufloat_t f_24 = cells_f_F[kap + 23*n_maxcells];
		ufloat_t f_25 = cells_f_F[kap + 26*n_maxcells];
		ufloat_t f_26 = cells_f_F[kap + 25*n_maxcells];
		ufloat_t rho_kap = +f_0 +f_1 +f_2 +f_3 +f_4 +f_5 +f_6 +f_7 +f_8 +f_9 +f_10 +f_11 +f_12 +f_13 +f_14 +f_15 +f_16 +f_17 +f_18 +f_19 +f_20 +f_21 +f_22 +f_23 +f_24 +f_25 +f_26;
		ufloat_t u_kap = ( +f_1 -f_2 +f_7 -f_8 +f_9 -f_10 +f_13 -f_14 +f_15 -f_16 +f_19 -f_20 +f_21 -f_22 +f_23 -f_24 -f_25 +f_26) / rho_kap;
		ufloat_t v_kap = ( +f_3 -f_4 +f_7 -f_8 +f_11 -f_12 -f_13 +f_14 +f_17 -f_18 +f_19 -f_20 +f_21 -f_22 -f_23 +f_24 +f_25 -f_26) / rho_kap;
		ufloat_t w_kap = ( +f_5 -f_6 +f_9 -f_10 +f_11 -f_12 -f_15 +f_16 -f_17 +f_18 +f_19 -f_20 -f_21 +f_22 +f_23 -f_24 +f_25 -f_26) / rho_kap;
		ufloat_t cdotu = N_Pf(0.0);
		ufloat_t udotu = u_kap*u_kap + v_kap*v_kap + w_kap*w_kap;
		ufloat_t omeg = otau;
		ufloat_t omegp = N_Pf(1.0) - omeg;
		
		// Collision step.
		cdotu = 0; f_0 = f_0*omegp + ( N_Pf(0.296296296296296)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		cdotu = +u_kap; f_1 = f_1*omegp + ( N_Pf(0.074074074074074)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		cdotu = -u_kap; f_2 = f_2*omegp + ( N_Pf(0.074074074074074)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		cdotu = +v_kap; f_3 = f_3*omegp + ( N_Pf(0.074074074074074)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		cdotu = -v_kap; f_4 = f_4*omegp + ( N_Pf(0.074074074074074)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		cdotu = +w_kap; f_5 = f_5*omegp + ( N_Pf(0.074074074074074)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		cdotu = -w_kap; f_6 = f_6*omegp + ( N_Pf(0.074074074074074)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		cdotu = +u_kap+v_kap; f_7 = f_7*omegp + ( N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		cdotu = -u_kap-v_kap; f_8 = f_8*omegp + ( N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		cdotu = +u_kap+w_kap; f_9 = f_9*omegp + ( N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		cdotu = -u_kap-w_kap; f_10 = f_10*omegp + ( N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		cdotu = +v_kap+w_kap; f_11 = f_11*omegp + ( N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		cdotu = -v_kap-w_kap; f_12 = f_12*omegp + ( N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		cdotu = +u_kap-v_kap; f_13 = f_13*omegp + ( N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		cdotu = -u_kap+v_kap; f_14 = f_14*omegp + ( N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		cdotu = +u_kap-w_kap; f_15 = f_15*omegp + ( N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		cdotu = -u_kap+w_kap; f_16 = f_16*omegp + ( N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		cdotu = +v_kap-w_kap; f_17 = f_17*omegp + ( N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		cdotu = -v_kap+w_kap; f_18 = f_18*omegp + ( N_Pf(0.018518518518519)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		cdotu = +u_kap+v_kap+w_kap; f_19 = f_19*omegp + ( N_Pf(0.004629629629630)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		cdotu = -u_kap-v_kap-w_kap; f_20 = f_20*omegp + ( N_Pf(0.004629629629630)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		cdotu = +u_kap+v_kap-w_kap; f_21 = f_21*omegp + ( N_Pf(0.004629629629630)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		cdotu = -u_kap-v_kap+w_kap; f_22 = f_22*omegp + ( N_Pf(0.004629629629630)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		cdotu = +u_kap-v_kap+w_kap; f_23 = f_23*omegp + ( N_Pf(0.004629629629630)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		cdotu = -u_kap+v_kap-w_kap; f_24 = f_24*omegp + ( N_Pf(0.004629629629630)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		cdotu = -u_kap+v_kap+w_kap; f_25 = f_25*omegp + ( N_Pf(0.004629629629630)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		cdotu = +u_kap-v_kap-w_kap; f_26 = f_26*omegp + ( N_Pf(0.004629629629630)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu) )*omeg;
		
		// Impose BC (only checking +y in 2D for the lid).
#if (N_CONN_TYPE==0)
		if (K_kap == Nx-1)
#else
		if (cells_ID_nbr[kap + 5*n_maxcells] < 0)
#endif
		{
			// p = 5.
			cdotu = N_Pf(0.000000000000000); f_5 = f_5 - N_Pf(2.0)*N_Pf(0.055555555555556)*N_Pf(3.0)*cdotu;
			
			// p = 9.
			cdotu = N_Pf(0.050000000000000); f_9 = f_9 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
			
			// p = 11.
			cdotu = N_Pf(0.000000000000000); f_11 = f_11 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
			
			// p = 16.
			cdotu = N_Pf(-0.050000000000000); f_16 = f_16 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
			
			// p = 18.
			cdotu = N_Pf(0.000000000000000); f_18 = f_18 - N_Pf(2.0)*N_Pf(0.027777777777778)*N_Pf(3.0)*cdotu;
			
			// p = 19.
			cdotu = N_Pf(0.050000000000000); f_19 = f_19 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
			
			// p = 22.
			cdotu = N_Pf(-0.050000000000000); f_22 = f_22 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
			
			// p = 23.
			cdotu = N_Pf(0.050000000000000); f_23 = f_23 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
			
			// p = 25.
			cdotu = N_Pf(-0.050000000000000); f_25 = f_25 - N_Pf(2.0)*N_Pf(0.004629629629630)*N_Pf(3.0)*cdotu;
		}
		
		// Store DDFs.
		cells_f_F[kap + 0*n_maxcells] = f_0;
		cells_f_F[kap + 1*n_maxcells] = f_1;
		cells_f_F[kap + 2*n_maxcells] = f_2;
		cells_f_F[kap + 3*n_maxcells] = f_3;
		cells_f_F[kap + 4*n_maxcells] = f_4;
		cells_f_F[kap + 5*n_maxcells] = f_5;
		cells_f_F[kap + 6*n_maxcells] = f_6;
		cells_f_F[kap + 7*n_maxcells] = f_7;
		cells_f_F[kap + 8*n_maxcells] = f_8;
		cells_f_F[kap + 9*n_maxcells] = f_9;
		cells_f_F[kap + 10*n_maxcells] = f_10;
		cells_f_F[kap + 11*n_maxcells] = f_11;
		cells_f_F[kap + 12*n_maxcells] = f_12;
		cells_f_F[kap + 13*n_maxcells] = f_13;
		cells_f_F[kap + 14*n_maxcells] = f_14;
		cells_f_F[kap + 15*n_maxcells] = f_15;
		cells_f_F[kap + 16*n_maxcells] = f_16;
		cells_f_F[kap + 17*n_maxcells] = f_17;
		cells_f_F[kap + 18*n_maxcells] = f_18;
		cells_f_F[kap + 19*n_maxcells] = f_19;
		cells_f_F[kap + 20*n_maxcells] = f_20;
		cells_f_F[kap + 21*n_maxcells] = f_21;
		cells_f_F[kap + 22*n_maxcells] = f_22;
		cells_f_F[kap + 23*n_maxcells] = f_23;
		cells_f_F[kap + 24*n_maxcells] = f_24;
		cells_f_F[kap + 25*n_maxcells] = f_25;
		cells_f_F[kap + 26*n_maxcells] = f_26;
#endif
	}
}

__global__
void Cu_Stream
(
	int n_maxcells,
	int *cells_ID_nbr, ufloat_t *cells_f_F
)
{
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	int I_kap = kap % Nx;
	int J_kap = (kap / Nx) % Nx;
#if (N_DIM==3)
	int K_kap = (kap / Nx) / Nx;
#endif
	
	ufloat_t f_p = N_Pf(0.0);
	ufloat_t f_pb = N_Pf(0.0);
	int nbr_b = -1;
	
	if (kap < n_maxcells)
	{		
#if (N_DIM==2)
		
		
		
		// p = 1, pb = 3
#if (N_CONN_TYPE==0)
		if (I_kap+1 < Nx)
		{
			nbr_b = (I_kap+1) + Nx*(J_kap);
#else
		nbr_b = cells_ID_nbr[kap + 1*n_maxcells];
		if (nbr_b >= 0)
		{
#endif
			f_p = cells_f_F[kap + 1*n_maxcells];
			f_pb = cells_f_F[nbr_b + 3*n_maxcells];
			
			cells_f_F[nbr_b + 3*n_maxcells] = f_p;
			cells_f_F[kap + 1*n_maxcells] = f_pb;
		}
		
		// p = 2, pb = 4
#if (N_CONN_TYPE==0)
		if (J_kap+1 < Nx)
		{
			nbr_b = (I_kap) + Nx*(J_kap+1);
#else
		nbr_b = cells_ID_nbr[kap + 2*n_maxcells];
		if (nbr_b >= 0)
		{
#endif
			f_p = cells_f_F[kap + 2*n_maxcells];
			f_pb = cells_f_F[nbr_b + 4*n_maxcells];
			
			cells_f_F[nbr_b + 4*n_maxcells] = f_p;
			cells_f_F[kap + 2*n_maxcells] = f_pb;
		}
		
		// p = 5, pb = 7
#if (N_CONN_TYPE==0)
		if (I_kap+1 < Nx && J_kap+1 < Nx)
		{
			nbr_b = (I_kap+1) + Nx*(J_kap+1);
#else
		nbr_b = cells_ID_nbr[kap + 5*n_maxcells];
		if (nbr_b >= 0)
		{
#endif
			f_p = cells_f_F[kap + 5*n_maxcells];
			f_pb = cells_f_F[nbr_b + 7*n_maxcells];
			
			cells_f_F[nbr_b + 7*n_maxcells] = f_p;
			cells_f_F[kap + 5*n_maxcells] = f_pb;
		}
		
		// p = 8, pb = 6
#if (N_CONN_TYPE==0)
		if (I_kap+1 < Nx && J_kap-1 >= 0)
		{
			nbr_b = (I_kap+1) + Nx*(J_kap-1);
#else
		nbr_b = cells_ID_nbr[kap + 8*n_maxcells];
		if (nbr_b >= 0)
		{
#endif
			f_p = cells_f_F[kap + 8*n_maxcells];
			f_pb = cells_f_F[nbr_b + 6*n_maxcells];
			
			cells_f_F[nbr_b + 6*n_maxcells] = f_p;
			cells_f_F[kap + 8*n_maxcells] = f_pb;
		}
		
		
		
#else // N_Q==19 or N_Q==27



		// p = 1, pb = 2
#if (N_CONN_TYPE==0)
		if (I_kap+1 < Nx)
		{
			nbr_b = (I_kap+1) + Nx*(J_kap) + Nx*Nx*(K_kap);
#else
		nbr_b = cells_ID_nbr[kap + 1*n_maxcells];
		if (nbr_b >= 0)
		{
#endif
			f_p = cells_f_F[kap + 1*n_maxcells];
			f_pb = cells_f_F[nbr_b + 2*n_maxcells];
			
			cells_f_F[nbr_b + 2*n_maxcells] = f_p;
			cells_f_F[kap + 1*n_maxcells] = f_pb;
		}
		
		// p = 3, pb = 4
#if (N_CONN_TYPE==0)
		if (J_kap+1 < Nx)
		{
			nbr_b = (I_kap) + Nx*(J_kap+1) + Nx*Nx*(K_kap);
#else
		nbr_b = cells_ID_nbr[kap + 3*n_maxcells];
		if (nbr_b >= 0)
		{
#endif
			f_p = cells_f_F[kap + 3*n_maxcells];
			f_pb = cells_f_F[nbr_b + 4*n_maxcells];
			
			cells_f_F[nbr_b + 4*n_maxcells] = f_p;
			cells_f_F[kap + 3*n_maxcells] = f_pb;
		}
		
		// p = 5, pb = 6
#if (N_CONN_TYPE==0)
		if (K_kap+1 < Nx)
		{
			nbr_b = (I_kap) + Nx*(J_kap) + Nx*Nx*(K_kap+1);
#else
		nbr_b = cells_ID_nbr[kap + 5*n_maxcells];
		if (nbr_b >= 0)
		{
#endif
			f_p = cells_f_F[kap + 5*n_maxcells];
			f_pb = cells_f_F[nbr_b + 6*n_maxcells];
			
			cells_f_F[nbr_b + 6*n_maxcells] = f_p;
			cells_f_F[kap + 5*n_maxcells] = f_pb;
		}
		
		// p = 7, pb = 8
#if (N_CONN_TYPE==0)
		if (I_kap+1 < Nx && J_kap+1 < Nx)
		{
			nbr_b = (I_kap+1) + Nx*(J_kap+1) + Nx*Nx*(K_kap);
#else
		nbr_b = cells_ID_nbr[kap + 7*n_maxcells];
		if (nbr_b >= 0)
		{
#endif
			f_p = cells_f_F[kap + 7*n_maxcells];
			f_pb = cells_f_F[nbr_b + 8*n_maxcells];
			
			cells_f_F[nbr_b + 8*n_maxcells] = f_p;
			cells_f_F[kap + 7*n_maxcells] = f_pb;
		}
		
		// p = 9, pb = 10
#if (N_CONN_TYPE==0)
		if (I_kap+1 < Nx && K_kap+1 < Nx)
		{
			nbr_b = (I_kap+1) + Nx*(J_kap) + Nx*Nx*(K_kap+1);
#else
		nbr_b = cells_ID_nbr[kap + 9*n_maxcells];
		if (nbr_b >= 0)
		{
#endif
			f_p = cells_f_F[kap + 9*n_maxcells];
			f_pb = cells_f_F[nbr_b + 10*n_maxcells];
			
			cells_f_F[nbr_b + 10*n_maxcells] = f_p;
			cells_f_F[kap + 9*n_maxcells] = f_pb;
		}
		
		// p = 11, pb = 12
#if (N_CONN_TYPE==0)
		if (J_kap+1 < Nx && K_kap+1 < Nx)
		{
			nbr_b = (I_kap) + Nx*(J_kap+1) + Nx*Nx*(K_kap+1);
#else
		nbr_b = cells_ID_nbr[kap + 11*n_maxcells];
		if (nbr_b >= 0)
		{
#endif
			f_p = cells_f_F[kap + 11*n_maxcells];
			f_pb = cells_f_F[nbr_b + 12*n_maxcells];
			
			cells_f_F[nbr_b + 12*n_maxcells] = f_p;
			cells_f_F[kap + 11*n_maxcells] = f_pb;
		}
		
		// p = 13, pb = 14
#if (N_CONN_TYPE==0)
		if (I_kap+1 < Nx && J_kap-1 >= 0)
		{
			nbr_b = (I_kap+1) + Nx*(J_kap-1) + Nx*Nx*(K_kap);
#else
		nbr_b = cells_ID_nbr[kap + 13*n_maxcells];
		if (nbr_b >= 0)
		{
#endif
			f_p = cells_f_F[kap + 13*n_maxcells];
			f_pb = cells_f_F[nbr_b + 14*n_maxcells];
			
			cells_f_F[nbr_b + 14*n_maxcells] = f_p;
			cells_f_F[kap + 13*n_maxcells] = f_pb;
		}
		
		// p = 15, pb = 16
#if (N_CONN_TYPE==0)
		if (I_kap+1 < Nx && K_kap-1 >= 0)
		{
			nbr_b = (I_kap+1) + Nx*(J_kap) + Nx*Nx*(K_kap-1);
#else
		nbr_b = cells_ID_nbr[kap + 15*n_maxcells];
		if (nbr_b >= 0)
		{
#endif
			f_p = cells_f_F[kap + 15*n_maxcells];
			f_pb = cells_f_F[nbr_b + 16*n_maxcells];
			
			cells_f_F[nbr_b + 16*n_maxcells] = f_p;
			cells_f_F[kap + 15*n_maxcells] = f_pb;
		}
		
		// p = 17, pb = 18
#if (N_CONN_TYPE==0)
		if (J_kap+1 < Nx && K_kap-1 >= 0)
		{
			nbr_b = (I_kap) + Nx*(J_kap+1) + Nx*Nx*(K_kap-1);
#else
		nbr_b = cells_ID_nbr[kap + 17*n_maxcells];
		if (nbr_b >= 0)
		{
#endif
			f_p = cells_f_F[kap + 17*n_maxcells];
			f_pb = cells_f_F[nbr_b + 18*n_maxcells];
			
			cells_f_F[nbr_b + 18*n_maxcells] = f_p;
			cells_f_F[kap + 17*n_maxcells] = f_pb;
		}
		
		
		
#if (N_Q==27) // N_Q==27



		// p = 19, pb = 20
#if (N_CONN_TYPE==0)
		if (I_kap+1 < Nx && J_kap+1 < Nx && K_kap+1 < Nx)
		{
			nbr_b = (I_kap+1) + Nx*(J_kap+1) + Nx*Nx*(K_kap+1);
#else
		nbr_b = cells_ID_nbr[kap + 19*n_maxcells];
		if (nbr_b >= 0)
		{
#endif
			f_p = cells_f_F[kap + 19*n_maxcells];
			f_pb = cells_f_F[nbr_b + 20*n_maxcells];
			
			cells_f_F[nbr_b + 20*n_maxcells] = f_p;
			cells_f_F[kap + 19*n_maxcells] = f_pb;
		}
		
		// p = 21, pb = 22
#if (N_CONN_TYPE==0)
		if (I_kap+1 < Nx && J_kap+1 < Nx && K_kap-1 >= 0)
		{
			nbr_b = (I_kap+1) + Nx*(J_kap+1) + Nx*Nx*(K_kap-1);
#else
		nbr_b = cells_ID_nbr[kap + 21*n_maxcells];
		if (nbr_b >= 0)
		{
#endif
			f_p = cells_f_F[kap + 21*n_maxcells];
			f_pb = cells_f_F[nbr_b + 22*n_maxcells];
			
			cells_f_F[nbr_b + 22*n_maxcells] = f_p;
			cells_f_F[kap + 21*n_maxcells] = f_pb;
		}
		
		// p = 23, pb = 24
#if (N_CONN_TYPE==0)
		if (I_kap+1 < Nx && J_kap-1 >= 0 && K_kap+1 < Nx)
		{
			nbr_b = (I_kap+1) + Nx*(J_kap-1) + Nx*Nx*(K_kap+1);
#else
		nbr_b = cells_ID_nbr[kap + 23*n_maxcells];
		if (nbr_b >= 0)
		{
#endif
			f_p = cells_f_F[kap + 23*n_maxcells];
			f_pb = cells_f_F[nbr_b + 24*n_maxcells];
			
			cells_f_F[nbr_b + 24*n_maxcells] = f_p;
			cells_f_F[kap + 23*n_maxcells] = f_pb;
		}
		
		// p = 26, pb = 25
#if (N_CONN_TYPE==0)
		if (I_kap+1 < Nx && J_kap-1 >= 0 && K_kap-1 >= 0)
		{
			nbr_b = (I_kap+1) + Nx*(J_kap-1) + Nx*Nx*(K_kap-1);
#else
		nbr_b = cells_ID_nbr[kap + 26*n_maxcells];
		if (nbr_b >= 0)
		{
#endif
			f_p = cells_f_F[kap + 26*n_maxcells];
			f_pb = cells_f_F[nbr_b + 25*n_maxcells];
			
			cells_f_F[nbr_b + 25*n_maxcells] = f_p;
			cells_f_F[kap + 26*n_maxcells] = f_pb;
		}
		
		
		
#endif

#endif
	}
}



int myrandom (int i)
{
	// From https://cplusplus.com/reference/algorithm/random_shuffle/
	
	return std::rand()%i;
}

int GCI(int i, int j, int k)
{
	// Stands for 'Get Connectivity Index'.
	
	return i + (Nx+2)*(j) + (Nx+2)*(Nx+2)*k;
}

int BuildConnectivity(int n_maxcells, int *cells_ID_nbr, int *Id_list)
{
#if (N_OCTREE==0)
	// Array with Ids in a row.
	for (int i = 0; i < n_maxcells; i++)
		Id_list[i] = i;
	
#if (N_SHUFFLE==1)
	// Scramble the Ids for one of the tests.
	std::random_shuffle(Id_list, Id_list + n_maxcells, myrandom);
#endif
	
	// Adjust domain height.
	int height = Nx;
	if (N_DIM==2)
		height = 1;
	
	// Fill the domain.
	int *domain = new int[(Nx+2)*(Nx+2)*(height+2)];
	for (int k = 0; k < height; k++)
	{
		for (int j = 0; j < Nx; j++)
		{
			for (int i = 0; i < Nx; i++)
			{
				domain[GCI(i+1,j+1,k+1)] = Id_list[i + Nx*j + Nx*Nx*k];
			}
		}
	}
#else
	// Array with Ids in a row.
	int N_CHILDREN = (N_DIM-1)*4;
	for (int i = 0; i < n_maxcells/N_CHILDREN; i++)
		Id_list[i] = i*N_CHILDREN;
	
#if (N_SHUFFLE==1)
	// Scramble the Ids for one of the tests.
	std::random_shuffle(Id_list, Id_list + n_maxcells/N_CHILDREN, myrandom);
#endif
	
	// Adjust domain height.
	int height = Nx;
	if (N_DIM==2)
		height = 1;
	
	// Fill the domain.
	int *domain = new int[(Nx+2)*(Nx+2)*(height+2)];
#if (N_DIM==2)
	for (int k = 0; k < height; k++)
	{
		for (int j = 0; j < Nx/2; j++)
		{
			for (int i = 0; i < Nx/2; i++)
			{
				int tmp = Id_list[i + (Nx/2)*j];
				domain[GCI(2*i+1 + 0,2*j+1 + 0,k+1 + 0)] = tmp + 0;
				domain[GCI(2*i+1 + 1,2*j+1 + 0,k+1 + 0)] = tmp + 1;
				domain[GCI(2*i+1 + 0,2*j+1 + 1,k+1 + 0)] = tmp + 2;
				domain[GCI(2*i+1 + 1,2*j+1 + 1,k+1 + 0)] = tmp + 3;
			}
		}
	}
#else
	for (int k = 0; k < Nx/2; k++)
	{
		for (int j = 0; j < Nx/2; j++)
		{
			for (int i = 0; i < Nx/2; i++)
			{
				int tmp = Id_list[i + (Nx/2)*j + (Nx/2)*(Nx/2)*k];
				domain[GCI(2*i+1 + 0,2*j+1 + 0,2*k+1 + 0)] = tmp + 0;
				domain[GCI(2*i+1 + 1,2*j+1 + 0,2*k+1 + 0)] = tmp + 1;
				domain[GCI(2*i+1 + 0,2*j+1 + 1,2*k+1 + 0)] = tmp + 2;
				domain[GCI(2*i+1 + 1,2*j+1 + 1,2*k+1 + 0)] = tmp + 3;
				domain[GCI(2*i+1 + 0,2*j+1 + 0,2*k+1 + 1)] = tmp + 4;
				domain[GCI(2*i+1 + 1,2*j+1 + 0,2*k+1 + 1)] = tmp + 5;
				domain[GCI(2*i+1 + 0,2*j+1 + 1,2*k+1 + 1)] = tmp + 6;
				domain[GCI(2*i+1 + 1,2*j+1 + 1,2*k+1 + 1)] = tmp + 7;
			}
		}
	}
#endif
	for (int k = 0; k < height; k++)
	{
		for (int j = 0; j < Nx; j++)
		{
			for (int i = 0; i < Nx; i++)
			{
				Id_list[i + Nx*j + Nx*Nx*k] = domain[GCI(i+1,j+1,k+1)];
			}
		}
	}
#endif
	
	// Set the domain boundary Ids up.
		// X
	for (int k = 0; k < height+2; k++)
	{
		for (int j = 0; j < Nx+2; j++)
		{	
			domain[GCI(0,j,k)] = -1;
			domain[GCI(Nx+2 - 1,j,k)] = -2;
		}
	}
		// Y
	for (int k = 0; k < height+2; k++)
	{
		for (int i = 0; i < Nx+2; i++)
		{
			domain[GCI(i,0,k)] = -3;
			domain[GCI(i,Nx+2 - 1,k)] = -4;
		}
	}
#if (N_DIM==3)
		// Z
	for (int j = 0; j < Nx+2; j++)
	{
		for (int i = 0; i < Nx+2; i++)
		{
			domain[GCI(i,j,0)] = -5;
			domain[GCI(i,j,Nx+2 - 1)] = -6;
		}
	}
#endif
	
	// Fill 'cells_ID_nbr'.
	for (int k = 0; k < height; k++)
	{
		for (int j = 0; j < Nx; j++)
		{
			for (int i = 0; i < Nx; i++)
			{
				//int kap = i + Nx*j + Nx*Nx*k;
				int kap = domain[GCI(i+1,j+1,k+1)];
#if (N_DIM==2)
				cells_ID_nbr[kap + 0*n_maxcells] = domain[GCI(i+1 + 0,j+1 + 0,k+1 + 0)];
				cells_ID_nbr[kap + 1*n_maxcells] = domain[GCI(i+1 + 1,j+1 + 0,k+1 + 0)];
				cells_ID_nbr[kap + 2*n_maxcells] = domain[GCI(i+1 + 0,j+1 + 1,k+1 + 0)];
				cells_ID_nbr[kap + 3*n_maxcells] = domain[GCI(i+1 + -1,j+1 + 0,k+1 + 0)];
				cells_ID_nbr[kap + 4*n_maxcells] = domain[GCI(i+1 + 0,j+1 + -1,k+1 + 0)];
				cells_ID_nbr[kap + 5*n_maxcells] = domain[GCI(i+1 + 1,j+1 + 1,k+1 + 0)];
				cells_ID_nbr[kap + 6*n_maxcells] = domain[GCI(i+1 + -1,j+1 + 1,k+1 + 0)];
				cells_ID_nbr[kap + 7*n_maxcells] = domain[GCI(i+1 + -1,j+1 + -1,k+1 + 0)];
				cells_ID_nbr[kap + 8*n_maxcells] = domain[GCI(i+1 + 1,j+1 + -1,k+1 + 0)];
#else // N_Q==19 or N_Q==27
				cells_ID_nbr[kap + 0*n_maxcells] = domain[GCI(i+1 + 0,j+1 + 0,k+1 + 0)];
				cells_ID_nbr[kap + 1*n_maxcells] = domain[GCI(i+1 + 1,j+1 + 0,k+1 + 0)];
				cells_ID_nbr[kap + 2*n_maxcells] = domain[GCI(i+1 + -1,j+1 + 0,k+1 + 0)];
				cells_ID_nbr[kap + 3*n_maxcells] = domain[GCI(i+1 + 0,j+1 + 1,k+1 + 0)];
				cells_ID_nbr[kap + 4*n_maxcells] = domain[GCI(i+1 + 0,j+1 + -1,k+1 + 0)];
				cells_ID_nbr[kap + 5*n_maxcells] = domain[GCI(i+1 + 0,j+1 + 0,k+1 + 1)];
				cells_ID_nbr[kap + 6*n_maxcells] = domain[GCI(i+1 + 0,j+1 + 0,k+1 + -1)];
				cells_ID_nbr[kap + 7*n_maxcells] = domain[GCI(i+1 + 1,j+1 + 1,k+1 + 0)];
				cells_ID_nbr[kap + 8*n_maxcells] = domain[GCI(i+1 + -1,j+1 + -1,k+1 + 0)];
				cells_ID_nbr[kap + 9*n_maxcells] = domain[GCI(i+1 + 1,j+1 + 0,k+1 + 1)];
				cells_ID_nbr[kap + 10*n_maxcells] = domain[GCI(i+1 + -1,j+1 + 0,k+1 + -1)];
				cells_ID_nbr[kap + 11*n_maxcells] = domain[GCI(i+1 + 0,j+1 + 1,k+1 + 1)];
				cells_ID_nbr[kap + 12*n_maxcells] = domain[GCI(i+1 + 0,j+1 + -1,k+1 + -1)];
				cells_ID_nbr[kap + 13*n_maxcells] = domain[GCI(i+1 + 1,j+1 + -1,k+1 + 0)];
				cells_ID_nbr[kap + 14*n_maxcells] = domain[GCI(i+1 + -1,j+1 + 1,k+1 + 0)];
				cells_ID_nbr[kap + 15*n_maxcells] = domain[GCI(i+1 + 1,j+1 + 0,k+1 + -1)];
				cells_ID_nbr[kap + 16*n_maxcells] = domain[GCI(i+1 + -1,j+1 + 0,k+1 + 1)];
				cells_ID_nbr[kap + 17*n_maxcells] = domain[GCI(i+1 + 0,j+1 + 1,k+1 + -1)];
				cells_ID_nbr[kap + 18*n_maxcells] = domain[GCI(i+1 + 0,j+1 + -1,k+1 + 1)];
#if (N_Q==27)
				cells_ID_nbr[kap + 19*n_maxcells] = domain[GCI(i+1 + 1,j+1 + 1,k+1 + 1)];
				cells_ID_nbr[kap + 20*n_maxcells] = domain[GCI(i+1 + -1,j+1 + -1,k+1 + -1)];
				cells_ID_nbr[kap + 21*n_maxcells] = domain[GCI(i+1 + 1,j+1 + 1,k+1 + -1)];
				cells_ID_nbr[kap + 22*n_maxcells] = domain[GCI(i+1 + -1,j+1 + -1,k+1 + 1)];
				cells_ID_nbr[kap + 23*n_maxcells] = domain[GCI(i+1 + 1,j+1 + -1,k+1 + 1)];
				cells_ID_nbr[kap + 24*n_maxcells] = domain[GCI(i+1 + -1,j+1 + 1,k+1 + -1)];
				cells_ID_nbr[kap + 25*n_maxcells] = domain[GCI(i+1 + -1,j+1 + 1,k+1 + 1)];
				cells_ID_nbr[kap + 26*n_maxcells] = domain[GCI(i+1 + 1,j+1 + -1,k+1 + -1)];
#endif
				
#endif
			}
		}
	}

	// Free allocations.
	delete[] domain;
	
	return 0;
}

int ComputeProperties(int kap, int n_maxcells, ufloat_t *cells_f_F, double *out)
{
#if (N_Q==9)
	ufloat_t f_0 = cells_f_F[kap + 0*n_maxcells];
	ufloat_t f_1 = cells_f_F[kap + 3*n_maxcells];
	ufloat_t f_2 = cells_f_F[kap + 4*n_maxcells];
	ufloat_t f_3 = cells_f_F[kap + 1*n_maxcells];
	ufloat_t f_4 = cells_f_F[kap + 2*n_maxcells];
	ufloat_t f_5 = cells_f_F[kap + 7*n_maxcells];
	ufloat_t f_6 = cells_f_F[kap + 8*n_maxcells];
	ufloat_t f_7 = cells_f_F[kap + 5*n_maxcells];
	ufloat_t f_8 = cells_f_F[kap + 6*n_maxcells];
	out[0] = (double)(+f_0 +f_1 +f_2 +f_3 +f_4 +f_5 +f_6 +f_7 +f_8);
	out[1] = (double)(( +f_1 -f_3 +f_5 -f_6 -f_7 +f_8) / out[0]);
	out[2] = (double)(( +f_2 -f_4 +f_5 +f_6 -f_7 -f_8) / out[0]);
	out[3] = 0.0;
#elif (N_Q==19)
	ufloat_t f_0 = cells_f_F[kap + 0*n_maxcells];
	ufloat_t f_1 = cells_f_F[kap + 2*n_maxcells];
	ufloat_t f_2 = cells_f_F[kap + 1*n_maxcells];
	ufloat_t f_3 = cells_f_F[kap + 4*n_maxcells];
	ufloat_t f_4 = cells_f_F[kap + 3*n_maxcells];
	ufloat_t f_5 = cells_f_F[kap + 6*n_maxcells];
	ufloat_t f_6 = cells_f_F[kap + 5*n_maxcells];
	ufloat_t f_7 = cells_f_F[kap + 8*n_maxcells];
	ufloat_t f_8 = cells_f_F[kap + 7*n_maxcells];
	ufloat_t f_9 = cells_f_F[kap + 10*n_maxcells];
	ufloat_t f_10 = cells_f_F[kap + 9*n_maxcells];
	ufloat_t f_11 = cells_f_F[kap + 12*n_maxcells];
	ufloat_t f_12 = cells_f_F[kap + 11*n_maxcells];
	ufloat_t f_13 = cells_f_F[kap + 14*n_maxcells];
	ufloat_t f_14 = cells_f_F[kap + 13*n_maxcells];
	ufloat_t f_15 = cells_f_F[kap + 16*n_maxcells];
	ufloat_t f_16 = cells_f_F[kap + 15*n_maxcells];
	ufloat_t f_17 = cells_f_F[kap + 18*n_maxcells];
	ufloat_t f_18 = cells_f_F[kap + 17*n_maxcells];
	out[0] = (double)(+f_0 +f_1 +f_2 +f_3 +f_4 +f_5 +f_6 +f_7 +f_8 +f_9 +f_10 +f_11 +f_12 +f_13 +f_14 +f_15 +f_16 +f_17 +f_18);
	out[1] = (double)(( +f_1 -f_2 +f_7 -f_8 +f_9 -f_10 +f_13 -f_14 +f_15 -f_16) / out[0]);
	out[2] = (double)(( +f_3 -f_4 +f_7 -f_8 +f_11 -f_12 -f_13 +f_14 +f_17 -f_18) / out[0]);
	out[3] = (double)(( +f_5 -f_6 +f_9 -f_10 +f_11 -f_12 -f_15 +f_16 -f_17 +f_18) / out[0]);
#else // N_Q==27
	ufloat_t f_0 = cells_f_F[kap + 0*n_maxcells];
	ufloat_t f_1 = cells_f_F[kap + 2*n_maxcells];
	ufloat_t f_2 = cells_f_F[kap + 1*n_maxcells];
	ufloat_t f_3 = cells_f_F[kap + 4*n_maxcells];
	ufloat_t f_4 = cells_f_F[kap + 3*n_maxcells];
	ufloat_t f_5 = cells_f_F[kap + 6*n_maxcells];
	ufloat_t f_6 = cells_f_F[kap + 5*n_maxcells];
	ufloat_t f_7 = cells_f_F[kap + 8*n_maxcells];
	ufloat_t f_8 = cells_f_F[kap + 7*n_maxcells];
	ufloat_t f_9 = cells_f_F[kap + 10*n_maxcells];
	ufloat_t f_10 = cells_f_F[kap + 9*n_maxcells];
	ufloat_t f_11 = cells_f_F[kap + 12*n_maxcells];
	ufloat_t f_12 = cells_f_F[kap + 11*n_maxcells];
	ufloat_t f_13 = cells_f_F[kap + 14*n_maxcells];
	ufloat_t f_14 = cells_f_F[kap + 13*n_maxcells];
	ufloat_t f_15 = cells_f_F[kap + 16*n_maxcells];
	ufloat_t f_16 = cells_f_F[kap + 15*n_maxcells];
	ufloat_t f_17 = cells_f_F[kap + 18*n_maxcells];
	ufloat_t f_18 = cells_f_F[kap + 17*n_maxcells];
	ufloat_t f_19 = cells_f_F[kap + 20*n_maxcells];
	ufloat_t f_20 = cells_f_F[kap + 19*n_maxcells];
	ufloat_t f_21 = cells_f_F[kap + 22*n_maxcells];
	ufloat_t f_22 = cells_f_F[kap + 21*n_maxcells];
	ufloat_t f_23 = cells_f_F[kap + 24*n_maxcells];
	ufloat_t f_24 = cells_f_F[kap + 23*n_maxcells];
	ufloat_t f_25 = cells_f_F[kap + 26*n_maxcells];
	ufloat_t f_26 = cells_f_F[kap + 25*n_maxcells];
	out[0] = (double)(+f_0 +f_1 +f_2 +f_3 +f_4 +f_5 +f_6 +f_7 +f_8 +f_9 +f_10 +f_11 +f_12 +f_13 +f_14 +f_15 +f_16 +f_17 +f_18 +f_19 +f_20 +f_21 +f_22 +f_23 +f_24 +f_25 +f_26);
	out[1] = (double)(( +f_1 -f_2 +f_7 -f_8 +f_9 -f_10 +f_13 -f_14 +f_15 -f_16 +f_19 -f_20 +f_21 -f_22 +f_23 -f_24 -f_25 +f_26) / out[0]);
	out[2] = (double)(( +f_3 -f_4 +f_7 -f_8 +f_11 -f_12 -f_13 +f_14 +f_17 -f_18 +f_19 -f_20 +f_21 -f_22 -f_23 +f_24 +f_25 -f_26) / out[0]);
	out[3] = (double)(( +f_5 -f_6 +f_9 -f_10 +f_11 -f_12 -f_15 +f_16 -f_17 +f_18 +f_19 -f_20 -f_21 +f_22 +f_23 -f_24 +f_25 -f_26) / out[0]);
#endif
		
	return 0;
}



/*
888b     d888          d8b          
8888b   d8888          Y8P          
88888b.d88888                       
888Y88888P888  8888b.  888 88888b.  
888 Y888P 888     "88b 888 888 "88b 
888  Y8P  888 .d888888 888 888  888 
888   "   888 888  888 888 888  888 
888       888 "Y888888 888 888  888 
*/



int main(int argc, char *argv[])
{
	// Establishes a square/cubic domain and lid-driven cavity boundary conditions.
	// Must specify Nx, N_Q, N_PRECISION, N_iters, N_CONN_TYPE, in 'confmake.sh'.
	
	
	
	// Base parameters.
	int		n_maxcells = 1;
	cudaStream_t	stream;
	ufloat_t	dx = N_Pf(1.0)/(ufloat_t)Nx;
	ufloat_t	tau = N_Pf(3.0)*N_Pf(v0) + N_Pf(0.5)*dx;
	ufloat_t	otau = dx/tau;
	
	
	// Pointers.
	ufloat_t 	*  cells_f_F;
	int		*  cells_ID_nbr;
	ufloat_t	*c_cells_f_F;
	int		*c_cells_ID_nbr;
	int		*Id_list;
	
	
	// Derived parameters.
	for (int i = 0; i < N_DIM; i++)
		n_maxcells *= Nx;
	
	
	// Initialization.
	cudaStreamCreate(&stream);
	cells_f_F = new ufloat_t[n_maxcells*N_Q];
	cells_ID_nbr = new int[n_maxcells*N_Q_max];
	Id_list = new int[n_maxcells];
	gpuErrchk( cudaMalloc((void **)&c_cells_f_F, n_maxcells*N_Q*sizeof(ufloat_t)) );
	gpuErrchk( cudaMalloc((void **)&c_cells_ID_nbr, n_maxcells*N_Q_max*sizeof(int)) );
	std::cout << "Initialized..." << std::endl;
	
	
	// Fill host arrays, send to GPU.
	BuildConnectivity(n_maxcells, cells_ID_nbr, Id_list);
	gpuErrchk( cudaMemcpy(c_cells_ID_nbr, cells_ID_nbr, n_maxcells*N_Q_max*sizeof(int), cudaMemcpyHostToDevice) );
	std::cout << "Built connectivity, sent to GPU..." << std::endl;
	
	
	// Solver.
	double ave_time_collide = 0.0;
	double ave_time_stream = 0.0;
	double tmp_time = 0.0;
	std::ofstream iter_time_out; iter_time_out.open("./out/time.txt");
	Cu_SetInitialConditions<<<(M_BLOCK+n_maxcells-1)/M_BLOCK, M_BLOCK, 0, stream>>>
	(
		n_maxcells,
		c_cells_f_F,
		N_Pf(1.0), N_Pf(0.0), N_Pf(0.0), N_Pf(0.0)
	);
	cudaDeviceSynchronize();
	for (int i = 0; i < N_iters; i++)
	{
		std::cout << "Iteration " << i << ", t = " << i*dx << std::endl;
		
		tic_simple("");
		Cu_Collide<<<(M_BLOCK+n_maxcells-1)/M_BLOCK, M_BLOCK, 0, stream>>>
		(
			n_maxcells, otau,
			c_cells_ID_nbr, c_cells_f_F
		);
		cudaDeviceSynchronize();
		tmp_time = toc_simple("\tCollide",T_US);
		ave_time_collide += tmp_time;
		iter_time_out << tmp_time << " ";
		
		tic_simple("");
		Cu_Stream<<<(M_BLOCK+n_maxcells-1)/M_BLOCK, M_BLOCK, 0, stream>>>
		(
			n_maxcells,
			c_cells_ID_nbr, c_cells_f_F
		);
		cudaDeviceSynchronize();
		tmp_time = toc_simple("\tStream",T_US);
		ave_time_stream += tmp_time;
		iter_time_out << tmp_time << std::endl;
	}
	cudaDeviceSynchronize();
	ave_time_collide /= (double)(N_iters);
	ave_time_stream /= (double)(N_iters);
	std::cout << "Averages | Collide: " << ave_time_collide << ", Stream: " << ave_time_stream << std::endl;
	std::cout << "Total: " << ave_time_collide + ave_time_stream << std::endl;
	iter_time_out.close();
	
	
	// Retrieve data from GPU.
	gpuErrchk( cudaMemcpy(cells_f_F, c_cells_f_F, n_maxcells*N_Q*sizeof(ufloat_t), cudaMemcpyDeviceToHost) );
	
	
	// Output. Using legacy 'Structured Points' VTK file format.
	std::cout << "Opened new VTK dataset..." << std::endl;
	std::ofstream output; output.open(std::string("./out/out_") + std::to_string(N_iters) + ".vtk");
	double u_kap[1+3];
	output << "# vtk DataFile Version 3.0\n";
	output << "vtk output\n";
	output << "ASCII\n";
	output << "DATASET STRUCTURED_POINTS\n";
#if (N_DIM==2)
	output << "DIMENSIONS " << Nx << " " << Nx << " " << 1 << "\n";
#else
	output << "DIMENSIONS " << Nx << " " << Nx << " " << Nx << "\n";
#endif
	output << "ORIGIN 0 0 0\n";
	output << "SPACING 1 1 1\n";
	output << "\n";
	
	output << "POINT_DATA " << n_maxcells << "\n";
	output << "SCALARS Density double 1\n";
	output << "LOOKUP_TABLE default\n";
	for (int kap_i = 0; kap_i < n_maxcells; kap_i++)
	{
		int kap = kap_i;
#if (N_CONN_TYPE==1)
		kap = Id_list[kap_i];
#endif
		
		ComputeProperties(kap, n_maxcells, cells_f_F, u_kap);
		output << u_kap[0] << std::endl;
	}
	output << "SCALARS CellId double 1\n";
	output << "LOOKUP_TABLE default\n";
	for (int kap_i = 0; kap_i < n_maxcells; kap_i++)
	{
		int kap = kap_i;
#if (N_CONN_TYPE==1)
		kap = Id_list[kap_i];
#endif
		
		output << kap << std::endl;
	}
	output << "\n";
	
	output << "VECTORS Velocity double\n";
	for (int kap_i = 0; kap_i < n_maxcells; kap_i++)
	{
		int kap = kap_i;
#if (N_CONN_TYPE==1)
		kap = Id_list[kap_i];
#endif
		
		ComputeProperties(kap, n_maxcells, cells_f_F, u_kap);
		output << u_kap[1] << " " << u_kap[2] << " " << u_kap[3] << std::endl;
	}
	output.close();
	std::cout << "Finished writing VTK dataset..." << std::endl;
	
	
	// Free allocations.
	delete[] cells_f_F;
	delete[] cells_ID_nbr;
	delete[] Id_list;
	gpuErrchk( cudaFree(c_cells_f_F) );
	gpuErrchk( cudaFree(c_cells_ID_nbr) );
	
	
	
	return 0;
}
