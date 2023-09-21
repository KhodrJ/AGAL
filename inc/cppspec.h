/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#ifndef CPP_SPEC_H
#define CPP_SPEC_H



// Solver parameters.
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
#ifndef VERBOSE
	#define VERBOSE 0						///< Controls level of output reporting.
#endif
#ifndef S_LES
	#define S_LES 0							///< Indicator for applying Smagorinsky subgrid-scale modeling.
#endif
#ifndef S_TYPE
	#define S_TYPE 1						///< Controls streaming type for Lattice Boltzmann solver.
									///< Value of 0 indicates naive streaming, 1 indicates in-place streaming.
#endif
#ifndef P_DIR_NAME
	#define P_DIR_NAME "../out/results/"
#endif



// Derived auxillary parameters.
#define N_DIMc (N_DIM-2)
#define N_U (1+N_DIM)
#define N_SKIPID -999
#define APPEND(x, y) x ## y
#if N_PRECISION == 1
	typedef double ufloat_t;
	#define N_Pf(x) (x)
#else
	typedef float ufloat_t;
	#define N_Pf(x) (APPEND(x,F))
#endif



// Number of children and cell-block size.
#if N_DIM == 2
	#define N_CHILDREN 4
	#define Nbx 4
	#define M_CBLOCK 16
	// Note: Changing between 16 and 64 requires changing: Interpolation (duh), Averaging (index computation).
#else
	#define N_CHILDREN 8
	#define Nbx 4
	#define M_CBLOCK 64
#endif



// GPU device parameters.
#define N_DEV 1								///< Number of GPU devices to employ.
#define M_BLOCK 128							///< Numer of threads per thread-block.
#define M_maxcells_roundoff 2048					///< Round-off parameter for computation of n_maxcells.



// Grid parameters.
#define Nx 64
#define Nx2 (Nx*Nx)
#define Nx3 (Nx*Nx*Nx)
#define Nbx2 (Nbx*Nbx)
#define Nbx3 (Nbx*Nbx*Nbx) 
#define C_smag N_Pf(0.2)
#define u_lid N_Pf(0.075)
#define L_c N_Pf(1.0)
#define Re_c 1000.00
#define v0 ((u_lid*L_c)/Re_c)
#define N_INIT_ITER 0



// Refinement parameters.
#define N_CONN_TYPE (1)



// Solver parameters.
#define S_INTERP_TYPE (0)



// Printing parameters.
#define P_REFINE (Nx)
#define P_PRINT (Nx*1000)
#define N_PRINT (1)
#define N_PRINT_LEVELS (MAX_LEVELS)
#define P_SHOW_ADVANCE (0)
#define P_PRINT_ADVANCE (0)
#define P_SHOW_REFINE (1)



// Miscellaneous.
#define CONV_B2GB 9.313225746154785e-10
#define T_S 0
#define T_MS 1
#define T_US 2



// General cpp shit.
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
    else {
        auto t_end = std::chrono::high_resolution_clock::now();
        //std::cout << "Elapsed time for process " << s << " is: " << (t_end-t_start).count()*1E-9 << " seconds\n";
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

inline double tic_simple(std::string s="", int mode=0, int scale=0, int print=1)
{
    static std::chrono::_V2::system_clock::time_point t_start_simple;
    
    if (mode==0)
	{
		t_start_simple = std::chrono::high_resolution_clock::now();
		return 0.0;
	}
    else {
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
