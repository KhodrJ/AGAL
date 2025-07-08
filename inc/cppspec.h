#ifndef CPPSPEC_H
#define CPPSPEC_H

// General C++ headers.
#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>
#include <map>
#include <chrono>
#include <algorithm>
#include <stdlib.h>
#include <cstdio>
#include <omp.h>
#include <vector>
#include <math.h>
#include <type_traits>

// Thrust.
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/gather.h>

// VTK.
#include "vtkAMRBox.h"
#include "vtkAMRUtilities.h"
#include "vtkCell.h"
#include "vtkCellData.h"
#include "vtkDataArray.h"
#include "vtkDoubleArray.h"
#include "vtkNew.h"
#include "vtkOverlappingAMR.h"
#include "vtkPointData.h"
#include "vtkPoints.h"
#include "vtkUniformGrid.h"
#include "vtkImageData.h"
#include "vtkXMLImageDataWriter.h"
#include "vtkXMLUniformGridAMRWriter.h"
#include "vtkCellDataToPointData.h"
#include "vtkContourFilter.h"
#include "vtkActor.h"
#include "vtkImageActor.h"
#include "vtkImageProperty.h"
#include "vtkImageCast.h"
#include "vtkLookupTable.h"
#include "vtkColorTransferFunction.h"
#include "vtkImageMapToColors.h"
#include "vtkImageMapper3D.h"
#include "vtkDataSetMapper.h"
#include "vtkCamera.h"
#include "vtkGraphicsFactory.h"
#include "vtkNamedColors.h"
#include "vtkPNGWriter.h"
#include "vtkPolyDataMapper.h"
#include "vtkProperty.h"
#include "vtkRenderWindow.h"
#include "vtkRenderer.h"
#include "vtkWindowToImageFilter.h"


struct ArgsPack
{
	// Base.
	const int N_DIM;
	const int N_DEV;
	const int Nqx;
	const int M_LBLOCK;
	const int M_LWBLOCK;
	const int M_BLOCK;
	const int M_RNDOFF;
	
	// Derived.
	const int N_CHILDREN             = N_DIM==2? 4:8;
	const int N_Q_max                = N_DIM==2? 9:27;
	const int N_QUADS                = N_DIM==2? Nqx*Nqx:Nqx*Nqx*Nqx;
	const int M_TBLOCK               = N_DIM==2? 16:64;
	const int M_WBLOCK               = N_DIM==2? 16:32;
	const int M_HBLOCK               = N_DIM==2? 36:216;
	const int M_CBLOCK               = N_QUADS*M_TBLOCK;
	
	constexpr ArgsPack(
		int N_DIM_=3,
		int M_BLOCK_=128,
		int N_DEV_=1,
		int Nqx_=1,
		int M_LBLOCK_=1,
		int M_LWBLOCK_=1,
		int M_RNDOFF_=2048
	) : 
		N_DIM(N_DIM_),
		M_BLOCK(M_BLOCK_),
		N_DEV(N_DEV_),
		Nqx(Nqx_),
		M_LBLOCK(M_LBLOCK_),
		M_LWBLOCK(M_LWBLOCK_),
		M_RNDOFF(M_RNDOFF_)
	{}
};

constexpr ArgsPack AP2D_DEF = ArgsPack(2); /// Default 2D argument pack.
constexpr ArgsPack AP3D_DEF = ArgsPack(3); /// Default 3D argument pack.

// constexpr int Nbx = 4;
constexpr int N_SYMMETRY = -998;
constexpr int N_SKIPID = -999;
constexpr double CONV_B2GB = 9.313225746154785e-10;
constexpr int T_S = 0;
constexpr int T_MS = 1;
constexpr int T_US = 2;

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

//! Reset values of GPU array.
/*! Reset the values of an array in GPU memory to a particular value.
    @param N is the length of the array.
    @param arr is a pointer to the array.
    @param val is the value being set.
*/
template<class T>
__global__
void Cu_ResetToValue(int N, T *arr, T val)
{
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (kap < N)
		arr[kap] = val;
}

//! Contract array elements by a specified amount.
/*! Retrieve elements of an array and store only a fraction of them (skipping over others) in a destination array not necessarily the same as the input.
    @param N is the length of the input array.
    @param arr is the input array.
    @param frac is the integer inverse of the fraction (i.e. to skip every N entries, frac should be set to N for a fraction of 1/N).
    @param arr2 is the destination array.
*/
template <class T, const ArgsPack *AP>
__global__
void Cu_ContractByFrac(int N, T *arr, int frac, T *arr2)
{
	constexpr int M_BLOCK = AP->M_BLOCK;
	__shared__ int s_arr[M_BLOCK];
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	int new_start = blockIdx.x*blockDim.x/frac;
	
	if (kap < N)
	{
		// Read arr into shared memory first.
		s_arr[threadIdx.x] = arr[kap];
		__syncthreads();
	
		if (threadIdx.x < M_BLOCK/frac)
		{
			arr2[new_start + threadIdx.x] = s_arr[frac*threadIdx.x];
		}
	}
}

//! Fill an array with a value equal to the index.
/*! @param N is the length of the input array.
    @param arr is the input array.
*/
template <class T>
__global__
void Cu_FillLinear(int N, T *arr)
{
	int kap = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (kap < N)
	{
		arr[kap] = kap;
	}
}

template <class T>
int DebugPrintDeviceArray(int N, T *d_arr)
{
	// Allocate memory for host array, copy device array data to it.
	T *h_arr = new T[N];
	gpuErrchk( cudaMemcpy(h_arr, d_arr, N*sizeof(T), cudaMemcpyDeviceToHost) );
	
	// Print data to console.
	std::cout << "DEBUG: ";
	for (int k = 0; k < N; k++)
		std::cout << h_arr[k] << " ";
	std::cout << std::endl;
	
	// Free memory of temporary host array.
	delete[] h_arr;
	
	return 0;
}

inline int DebugDrawCubeInMATLAB(std::ofstream &out, double x0, double x1, double y0, double y1, double z0, double z1, double c0, double c1, double c2)
{
	out << "plot3([" << x0 << " " << x1 << "],[" << y0 << " " << y0 << "],[" << z0 << " " << z0 << "],'Color',[" << c0 << " " << c1 << " " << c2 << "]);\n";
	out << "plot3([" << x0 << " " << x0 << "],[" << y0 << " " << y1 << "],[" << z0 << " " << z0 << "],'Color',[" << c0 << " " << c1 << " " << c2 << "]);\n";
	out << "plot3([" << x0 << " " << x1 << "],[" << y1 << " " << y1 << "],[" << z0 << " " << z0 << "],'Color',[" << c0 << " " << c1 << " " << c2 << "]);\n";
	out << "plot3([" << x1 << " " << x1 << "],[" << y0 << " " << y1 << "],[" << z0 << " " << z0 << "],'Color',[" << c0 << " " << c1 << " " << c2 << "]);\n";
	
	out << "plot3([" << x0 << " " << x1 << "],[" << y0 << " " << y0 << "],[" << z1 << " " << z1 << "],'Color',[" << c0 << " " << c1 << " " << c2 << "]);\n";
	out << "plot3([" << x0 << " " << x0 << "],[" << y0 << " " << y1 << "],[" << z1 << " " << z1 << "],'Color',[" << c0 << " " << c1 << " " << c2 << "]);\n";
	out << "plot3([" << x0 << " " << x1 << "],[" << y1 << " " << y1 << "],[" << z1 << " " << z1 << "],'Color',[" << c0 << " " << c1 << " " << c2 << "]);\n";
	out << "plot3([" << x1 << " " << x1 << "],[" << y0 << " " << y1 << "],[" << z1 << " " << z1 << "],'Color',[" << c0 << " " << c1 << " " << c2 << "]);\n";
	
	out << "plot3([" << x0 << " " << x0 << "],[" << y0 << " " << y0 << "],[" << z0 << " " << z1 << "],'Color',[" << c0 << " " << c1 << " " << c2 << "]);\n";
	out << "plot3([" << x0 << " " << x0 << "],[" << y1 << " " << y1 << "],[" << z0 << " " << z1 << "],'Color',[" << c0 << " " << c1 << " " << c2 << "]);\n";
	out << "plot3([" << x1 << " " << x1 << "],[" << y0 << " " << y0 << "],[" << z0 << " " << z1 << "],'Color',[" << c0 << " " << c1 << " " << c2 << "]);\n";
	out << "plot3([" << x1 << " " << x1 << "],[" << y1 << " " << y1 << "],[" << z0 << " " << z1 << "],'Color',[" << c0 << " " << c1 << " " << c2 << "]);\n";
	
	return 0;
}

inline int DebugDrawTriangleInMATLAB(std::ofstream &out, double vx1, double vy1, double vz1, double vx2, double vy2, double vz2, double vx3, double vy3, double vz3, double c0, double c1, double c2)
{
	out << "fill3([" << vx1 << " " << vx2 << " " << vx3 << " " << vx1 << "],[" << vy1 << " " << vy2 << " " << vy3 << " " << vy1 << "],[" << vz1 << " " << vz2 << " " << vz3 << " " << vz1 << "],[" << c0 << " " << c1 << " " << c2 << "]);\n";
	
	return 0;
}

template <class T> __device__ __forceinline__ T Tabs(T a);
template <> __device__ __forceinline__ int Tabs(int a) { return abs(a); }
template <> __device__ __forceinline__  float Tabs(float a) { return fabsf(a); }
template <> __device__ __forceinline__  double Tabs(double a) { return fabs(a); }

template <class T> __device__ __forceinline__ T Tpow(T a, T b);
template <> __device__ __forceinline__  float Tpow(float a, float b) { return powf(a,b); }
template <> __device__ __forceinline__  double Tpow(double a, double b) { return pow(a,b); }

template <class T> __device__ __forceinline__ T Tsqrt(T a);
template <> __device__ __forceinline__  float Tsqrt(float a) { return sqrtf(a); }
template <> __device__ __forceinline__  double Tsqrt(double a) { return sqrt(a); }

template <class T> __device__ __forceinline__ T Tacos(T a);
template <> __device__ __forceinline__  float Tacos(float a) { return acosf(a); }
template <> __device__ __forceinline__  double Tacos(double a) { return acos(a); }

#endif
