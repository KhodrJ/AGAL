/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

/* Parameters to configure before running.
 * 
 * N_DIM
 * N_PRECISION
 * MAX_LEVELS
 * P_DIR_NAME
 * Nx
 * 
 * N_PRINT
 * P_PRINT
 * N_PRINT_LEVELS
 * 
 * D2(/3)Q9(/19/27)
 * PERIODIC_X(/Y/Z)
 * 
 */

#ifndef CPP_SPEC_H
#define CPP_SPEC_H



/*
         8888888b.                                                 888                                      
         888   Y88b                                                888                                      
         888    888                                                888                                      
         888   d88P 8888b.  888d888 8888b.  88888b.d88b.   .d88b.  888888 .d88b.  888d888 .d8888b           
         8888888P"     "88b 888P"      "88b 888 "888 "88b d8P  Y8b 888   d8P  Y8b 888P"   88K               
         888       .d888888 888    .d888888 888  888  888 88888888 888   88888888 888     "Y8888b.          
         888       888  888 888    888  888 888  888  888 Y8b.     Y88b. Y8b.     888          X88          
88888888 888       "Y888888 888    "Y888888 888  888  888  "Y8888   "Y888 "Y8888  888      88888P' 88888888                                                                                                       
*/                                                                                         



// Input parameters.
#ifndef N_PRECISION
	#define N_PRECISION 0						///< Floating-point precision for data storage.
									///< 0 indicates single precision, 1 indicates double precision.
#endif
#ifndef N_DIM
	#define N_DIM 3							///< Number of dimensions.
#endif
#ifndef MAX_LEVELS
	#define MAX_LEVELS 3						///< Maximum number of levels in the grid hierarchy.
#endif
#ifndef MAX_LEVELS_INTERIOR
	#define MAX_LEVELS_INTERIOR MAX_LEVELS				///< Maximum number of levels for near-wall refinement.
#endif
#ifndef N_LEVEL_START
	#define N_LEVEL_START 0						///< Level to be counted as 'coarsest' level.
#endif
#ifndef S_TYPE
	#define S_TYPE 1						///< Controls streaming type for Lattice Boltzmann solver.
									///< Value of 0 indicates naive streaming, 1 indicates in-place streaming.
#endif
#ifndef S_INTERP_TYPE
	#define S_INTERP_TYPE 1						///< Interpolation type (0 - poly., 1 - weighted)
#endif
#ifndef S_INIT_TYPE
	#define S_INIT_TYPE 0						///< Controls initialization strategy.
									///< 0 indicates naive init., 1 indicates Mei's strategy.
#endif
#ifndef L_ci
	#define L_c N_Pf(1.0)
	#define L_fy N_Pf(1.0)
	#define L_fz N_Pf(1.0)
#else
	#define L_c N_Pf(L_ci)
	#define L_fy N_Pf(L_fyi)
	#define L_fz N_Pf(L_fzi)
#endif
#ifndef v0i
	#define v0 (N_Pf(1.5625E-5))
#else
	#define v0 (v0i)
#endif
#ifndef Nx
	#define Nx (256)						///< Size of coarse grid.
#endif
#ifndef B_TYPE
	#define B_TYPE 0
#endif
#ifndef P_DIR_NAME
	#define P_DIR_NAME "../out/results/"				
#endif
#ifndef N_CASE
	#define N_CASE (0)
#endif
	// Turbulence parameters.
#ifndef S_LES
	#define S_LES (0)						///< Indicator for applying Smagorinsky subgrid-scale modeling.
#endif
	// Refinement parameters.
#ifndef P_REFINEi
	#define P_REFINE (16)						///< Number of iterations between refinement/coarsening calls.
#else
	#define P_REFINE (P_REFINEi)
#endif
#ifndef N_REFINE_START
	#define N_REFINE_START (-4)					///< Arbitrary constant for refinement criterion log. comparison (any integer).
#endif
#ifndef N_REFINE_INC
	#define N_REFINE_INC (1)					///< Arbitrary constant for refinement criterion log. comparison (integer >=1).
#endif
#ifndef N_CONN_TYPE
	#define N_CONN_TYPE (1)						///< Switch for types of connectivity updates. Keep at 1.
#endif
#ifndef P_SHOW_REFINE
	#define P_SHOW_REFINE (1)					///< Show refinement execution times in terminal?
#endif
	// Probing parameters.
#ifndef N_PROBE
	#define N_PROBE (1)						///< Indicates whether or not to check for convergence via probing.
#endif
#ifndef N_PROBE_DENSITY
	#define N_PROBE_DENSITY (4)					///< Number of probe locations along an axis.
#endif
#ifndef N_PROBE_FREQUENCYi
	#define N_PROBE_FREQUENCY (Nx)					///< Number of iterations between check for convergence.
#else
	#define N_PROBE_FREQUENCY (N_PROBE_FREQUENCYi)
#endif
#ifndef V_PROBE_TOL
	#define V_PROBE_TOL (1e-5)					///< Tolerance for convergence via probing.
#endif
#ifndef N_PROBE_FORCE
	#define N_PROBE_FORCE (1)					///< Temporary indicator for reporting lift and drag forces.
#endif
#ifndef N_PROBE_F_FREQUENCYi
	#define N_PROBE_F_FREQUENCY (16)				///< Frequency of force report.
#else
	#define N_PROBE_F_FREQUENCY (N_PROBE_F_FREQUENCYi)
#endif
#ifndef N_PROBE_AVE
	#define N_PROBE_AVE (0)
#endif
#ifndef N_PROBE_AVE_FREQUENCYi
	#define N_PROBE_AVE_FREQUENCY (4)
	#define N_PROBE_AVE_START (100*Nx)
#else
	#define N_PROBE_AVE_FREQUENCY (N_PROBE_AVE_FREQUENCYi)
	#define N_PROBE_AVE_START (N_PROBE_AVE_STARTi)
#endif
	// Printing parameters.
#ifndef P_PRINTi
	#define P_PRINT (100*Nx)					///< Number of iterations between prints.
#else
	#define P_PRINT (P_PRINTi)
#endif
#ifndef N_PRINT
	#define N_PRINT (1)						///< Total number of prints to perform.
#endif
#ifndef N_PRINT_LEVELS
	#define N_PRINT_LEVELS (3)					///< Number of grid levels to include in print.
#endif
#ifndef P_SHOW_ADVANCE
	#define P_SHOW_ADVANCE (0)					///< Show advancement execution times in terminal?
#endif
#ifndef P_PRINT_ADVANCE
	#define P_PRINT_ADVANCE (0)						///< Number of iterations between advancement time terminal outputs.
#endif
	
	
	
// Derived auxillary parameters.
#define N_DIMc (N_DIM-2)						///< Number of dimensions minus 2.
#define N_U (1+N_DIM)							///< Number of macroscopic variables.
#define N_SKIPID -999							///< Value indicating to skip the operation for the current ID.
#define APPEND(x, y) x ## y
#if (N_PRECISION==1)
	typedef double ufloat_t;
	#define N_Pf(x) (x)
#else
	typedef float ufloat_t;
	#define N_Pf(x) (APPEND(x,f))
#endif

// Number of children and cell-block size.
#if N_DIM == 2
	#define N_CHILDREN 4						///< Number of children per block.
	#define Nbx 4							///< Number of cells along one axis in block.
	#define M_CBLOCK 16						///< Number of cells per block.
	// Note: Changing between 16 and 64 requires changing: Interpolation (duh), Averaging (index computation).
#else
	#define N_CHILDREN 8						///< Number of children per block.
	#define Nbx 4							///< Number of cells along one axis in block.
	#define M_CBLOCK 64						///< Number of cells per block.
#endif

// GPU device parameters.
#define N_DEV 1								///< Number of GPU devices to employ.
#define M_BLOCK 128							///< Numer of threads per thread-block.
#define M_maxcells_roundoff 2048					///< Round-off parameter for computation of n_maxcells.

// Miscellaneous.
#define CONV_B2GB 9.313225746154785e-10
#define T_S 0
#define T_MS 1
#define T_US 2



/*
         888    888                        888                                    
         888    888                        888                                    
         888    888                        888                                    
         8888888888  .d88b.   8888b.   .d88888  .d88b.  888d888 .d8888b           
         888    888 d8P  Y8b     "88b d88" 888 d8P  Y8b 888P"   88K               
         888    888 88888888 .d888888 888  888 88888888 888     "Y8888b.          
         888    888 Y8b.     888  888 Y88b 888 Y8b.     888          X88          
88888888 888    888  "Y8888  "Y888888  "Y88888  "Y8888  888      88888P' 88888888 
*/



// General cpp headers.
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <ctime>
#include <chrono>
#include <thread>
#include <unistd.h>
#include <algorithm>
#include <random>

// Thrust
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/gather.h>

// Block-based AMR writer.
#include "vtkAMRBox.h"
#include "vtkAMRUtilities.h"
#include "vtkCell.h"
#include "vtkCellData.h"
#include "vtkCompositeDataWriter.h"
#include "vtkDataArray.h"
#include "vtkDoubleArray.h"
#include "vtkMultiBlockDataSet.h"
#include "vtkNew.h"
#include "vtkOverlappingAMR.h"
#include "vtkPointData.h"
#include "vtkPoints.h"
#include "vtkUniformGrid.h"
#include "vtkXMLImageDataWriter.h"
#include "vtkXMLMultiBlockDataWriter.h"
#include "vtkXMLUniformGridAMRReader.h"
#include "vtkUnstructuredGrid.h"
#include "vtkXMLMultiBlockDataWriter.h"
#include "vtkXMLHierarchicalBoxDataWriter.h"
#include "vtkXMLUniformGridAMRWriter.h"



/*
         8888888888                         888    d8b                                     
         888                                888    Y8P                                     
         888                                888                                            
         8888888 888  888 88888b.   .d8888b 888888 888  .d88b.  88888b.  .d8888b           
         888     888  888 888 "88b d88P"    888    888 d88""88b 888 "88b 88K               
         888     888  888 888  888 888      888    888 888  888 888  888 "Y8888b.          
         888     Y88b 888 888  888 Y88b.    Y88b.  888 Y88..88P 888  888      X88          
88888888 888      "Y88888 888  888  "Y8888P  "Y888 888  "Y88P"  888  888  88888P' 88888888 
*/
                                                                                           
                                                                                           

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



// Prints out execution time measured with system clock.
// Credit: https://stackoverflow.com/a/63391159 (username: mma)
inline double tic(std::string s="", int mode=0, int scale=0)
{
	static std::chrono::_V2::system_clock::time_point t_start;
    
	if (mode==0)
	{
		t_start = std::chrono::high_resolution_clock::now();
		if (s != "")
			std::cout << "PROCESS: " << s << std::endl;
		return 0.0;
	}
	else
	{
		auto t_end = std::chrono::high_resolution_clock::now();
		if (scale == T_S)
		{
			std::cout << "END: Elapsed time for process " << s << " is: " << std::chrono::duration_cast<std::chrono::seconds>(t_end-t_start).count() << " seconds\n\n";
			return std::chrono::duration_cast<std::chrono::seconds>(t_end-t_start).count();
		}
		else if (scale == T_MS)
		{
			std::cout << "END: Elapsed time for process " << s << " is: " << std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count() << " milliseconds\n\n";
			return std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count();
		}
		else
		{
			std::cout << "END: Elapsed time for process " << s << " is: " << std::chrono::duration_cast<std::chrono::microseconds>(t_end-t_start).count() << " microseconds\n\n";
			return std::chrono::duration_cast<std::chrono::microseconds>(t_end-t_start).count();
		}
	}
}
inline double toc(std::string s, int scale=T_S) { return tic(s,1,scale); }



// Another version of tic-toc for separate measurements, controls whether to output to terminal.
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



#endif
