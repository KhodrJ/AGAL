/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#include "output.h"

#define Nbx 4

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
int VelGrad(int i, int j, int k, int *Nxi_f, double dx, int vol, double *data, double *out, Eigen::Matrix3f *m_velgrad)
{
	// Definitions.
	double u = data[(i+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*k) + 1*vol];
	double u_xM = 0.0;
	double u_xP = 0.0;
	double u_yM = 0.0;
	double u_yP = 0.0;
	double u_zM = 0.0;
	double u_zP = 0.0;
	double v = data[(i+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*k) + 2*vol];
	double v_xM = 0.0;
	double v_xP = 0.0;
	double v_yM = 0.0;
	double v_yP = 0.0;
	double v_zM = 0.0;
	double v_zP = 0.0;
	double w = data[(i+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*k) + 3*vol];
	double w_xM = 0.0;
	double w_xP = 0.0;
	double w_yM = 0.0;
	double w_yP = 0.0;
	double w_zM = 0.0;
	double w_zP = 0.0;
	
	// Corrections.
	if (i == 0)
	{
		u_xM = 2*u - data[((i+1)+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*k) + 1*vol];
		v_xM = 2*v - data[((i+1)+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*k) + 2*vol];
		w_xM = 2*w - data[((i+1)+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*k) + 3*vol];
	}
	if (i > 0)
	{
		u_xM = data[((i-1)+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*k) + 1*vol];
		v_xM = data[((i-1)+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*k) + 2*vol];
		w_xM = data[((i-1)+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*k) + 3*vol];
	}
	if (i == Nxi_f[0]-1)
	{
		u_xP = 2*u - data[((i-1)+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*k) + 1*vol];
		v_xP = 2*v - data[((i-1)+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*k) + 2*vol];
		w_xP = 2*w - data[((i-1)+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*k) + 3*vol];
	}
	if (i < Nxi_f[0]-1)
	{
		u_xP = data[((i+1)+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*k) + 1*vol];
		v_xP = data[((i+1)+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*k) + 2*vol];
		w_xP = data[((i+1)+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*k) + 3*vol];
	}
	if (j == 0)
	{
		u_yM = 2*u - data[(i+Nxi_f[0]*(j+1)+Nxi_f[0]*Nxi_f[1]*k) + 1*vol];
		v_yM = 2*v - data[(i+Nxi_f[0]*(j+1)+Nxi_f[0]*Nxi_f[1]*k) + 2*vol];
		w_yM = 2*w - data[(i+Nxi_f[0]*(j+1)+Nxi_f[0]*Nxi_f[1]*k) + 3*vol];
	}
	if (j > 0)
	{
		u_yM = data[(i+Nxi_f[0]*(j-1)+Nxi_f[0]*Nxi_f[1]*k) + 1*vol];
		v_yM = data[(i+Nxi_f[0]*(j-1)+Nxi_f[0]*Nxi_f[1]*k) + 2*vol];
		w_yM = data[(i+Nxi_f[0]*(j-1)+Nxi_f[0]*Nxi_f[1]*k) + 3*vol];
	}
	if (j == Nxi_f[1]-1)
	{
		u_yP = 2*u - data[(i+Nxi_f[0]*(j-1)+Nxi_f[0]*Nxi_f[1]*k) + 1*vol];
		v_yP = 2*v - data[(i+Nxi_f[0]*(j-1)+Nxi_f[0]*Nxi_f[1]*k) + 2*vol];
		w_yP = 2*w - data[(i+Nxi_f[0]*(j-1)+Nxi_f[0]*Nxi_f[1]*k) + 3*vol];
	}
	if (j < Nxi_f[1]-1)
	{
		u_yP = data[(i+Nxi_f[0]*(j+1)+Nxi_f[0]*Nxi_f[1]*k) + 1*vol];
		v_yP = data[(i+Nxi_f[0]*(j+1)+Nxi_f[0]*Nxi_f[1]*k) + 2*vol];
		w_yP = data[(i+Nxi_f[0]*(j+1)+Nxi_f[0]*Nxi_f[1]*k) + 3*vol];
	}
	if (Nxi_f[2]>1)
	{
		if (k == 0)
		{
			u_zM = 2*u - data[(i+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*(k+1)) + 1*vol];
			v_zM = 2*v - data[(i+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*(k+1)) + 2*vol];
			w_zM = 2*w - data[(i+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*(k+1)) + 3*vol];
		}
		if (k > 0)
		{
			u_zM = data[(i+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*(k-1)) + 1*vol];
			v_zM = data[(i+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*(k-1)) + 2*vol];
			w_zM = data[(i+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*(k-1)) + 3*vol];
		}
		if (k == Nxi_f[2]-1)
		{
			u_zP = 2*u - data[(i+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*(k-1)) + 1*vol];
			v_zP = 2*v - data[(i+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*(k-1)) + 2*vol];
			w_zP = 2*w - data[(i+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*(k-1)) + 3*vol];
		}
		if (k < Nxi_f[2]-1)
		{
			u_zP = data[(i+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*(k+1)) + 1*vol];
			v_zP = data[(i+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*(k+1)) + 2*vol];
			w_zP = data[(i+Nxi_f[0]*j+Nxi_f[0]*Nxi_f[1]*(k+1)) + 3*vol];
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

int main(int argc, char *argv[])
{
	// Parameters.
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
	long int      N_PROCS          = omp_get_max_threads();
	long int      buffer_length    = N_PROCS*1024*1024*1024;   // Default 1 GB buffer length for each process/thread.
	std::string   dirname          = "";
	std::string   filename         = "";
	if (argc < 2)
	{
// 		#pragma omp parallel
//                 {
//                         printf("Hello from process: %d / %d\n", omp_get_thread_num(), omp_get_max_threads());
//                 }
		
		std::cout << buffer_length << std::endl;
		std::cout << "Please supply the input directory with the direct-output file..." << std::endl;
		return 1;
	}
	else
	{
		// Set directory name. Ensure it ends with '/'.
		dirname = argv[1];
		if (dirname[dirname.length()-1] != '/')
			dirname = dirname + std::string("/");
		filename = dirname + std::string("out_direct.dat");
		std::cout << "Using " << filename << std::endl;
	}
	
	
	// Initial metadata.
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
	for (int d = 0; d < N_DIM; d++)
		Nxi_f[d] *= mult;
	vol = Nxi_f[0]*Nxi_f[1]*Nxi_f[2];
	std::cout << "    Volume of frames: " << vol << std::endl;
	
	
	// Start processing direct-output file.
	for (int K = 0; K < N_OUTPUT/N_PROCS+1; K++)
	{
		// Read density and velocity data for all threads.
		output_file.read(&buffer[0], N_PROCS*(3+1+2)*vol*sizeof(double));
		#pragma omp parallel
		{
		
		int t = omp_get_thread_num();
		int Kt = omp_get_thread_num() + K*N_PROCS;
		if (Kt < N_OUTPUT)
		{
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
		double *tmp_data = new double[n_data*vol];
		double *tmp_data_b = new double[n_data*vol];
		for (long int p = 0; p < n_data*vol; p++)
			tmp_data[p] = -1.0;
		for (int d = 0; d < 3+1; d++)
		{
			//output_file.read(&buffer[0], vol*sizeof(double));
			memcpy(&tmp_data[d*vol], &buffer[d*vol*sizeof(double) + t*(3+1+2)*vol*sizeof(double)], vol*sizeof(double));
		}
		//output_file.read(&buffer[], vol*sizeof(double));
		memcpy(&tmp_data[9*vol], &buffer[4*vol*sizeof(double) + t*(3+1+2)*vol*sizeof(double)], vol*sizeof(double));
		//output_file.read(&buffer[0], vol*sizeof(double));
		memcpy(&tmp_data[10*vol], &buffer[5*vol*sizeof(double) + t*(3+1+2)*vol*sizeof(double)], vol*sizeof(double));
		
		
		// Smoothing for better rendering.
		int n_smooths = 0;
		//int n_smooths = mult*Nbx;
		if (n_smooths > 0)
			std::cout << "[-] Smoothing grid..." << std::endl;
		//#pragma omp parallel for
		for (int kap = 0; kap < vol; kap++)
		{
			for (int p = 0; p < N_DIM+1; p++)
				tmp_data_b[kap + p*vol] = tmp_data[kap + p*vol];
		}
		for (int l = 0; l < n_smooths; l++)
		{
			std::cout << "    Smoothing iteration " << l << "..." << std::endl;
			
			if (N_DIM == 2)
			{
				//#pragma omp parallel for
				for (int j = 1; j < Nxi_f[1]-1; j++)
				{
					for (int i = 1; i < Nxi_f[0]-1; i++)
					{
						int kap = (i) + Nxi_f[0]*(j);
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
			else
			{
				//#pragma omp parallel for
				for (int k = 1; k < Nxi_f[2]-1; k++)
				{
					for (int j = 1; j < Nxi_f[1]-1; j++)
					{
						for (int i = 1; i < Nxi_f[0]-1; i++)
						{
							int kap = (i) + Nxi_f[0]*(j) + Nxi_f[0]*Nxi_f[1]*(k);
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
			
			//#pragma omp parallel for
			for (int kap = 0; kap < vol; kap++)
			{
				for (int p = 0; p < N_DIM+1; p++)
					tmp_data[kap + p*vol] = tmp_data_b[kap + p*vol];
			}
		}
		if (n_smooths > 0)
			std::cout << "    Finished smoothing grid..." << std::endl;
		
		
		// Compute remaining properties (e.g., vorticity, vector magnitudes...).
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
					VelGrad(i,j,k,   Nxi_f,dx0,vol,tmp_data,vel_grad,&m_velgrad);
// 					tmp_data[kap+4*vol] = 0.0;
// 					tmp_data[kap+5*vol] = 0.0;
// 					if (N_DIM == 3)
// 					{
					tmp_data[kap+4*vol] = vel_grad[2+3*1] - vel_grad[1+3*2];
					tmp_data[kap+5*vol] = vel_grad[0+3*2] - vel_grad[2+3*0];
// 					}
					tmp_data[kap+6*vol] = vel_grad[1+3*0] - vel_grad[0+3*1];
					
					// Velocity and vorticity magnitudes.
					tmp_data[kap+7*vol] = sqrt(tmp_data[kap+1*vol]*tmp_data[kap+1*vol] + tmp_data[kap+2*vol]*tmp_data[kap+2*vol] + tmp_data[kap+3*vol]*tmp_data[kap+3*vol]);
					tmp_data[kap+8*vol] = sqrt(tmp_data[kap+4*vol]*tmp_data[kap+4*vol] + tmp_data[kap+5*vol]*tmp_data[kap+5*vol] + tmp_data[kap+6*vol]*tmp_data[kap+6*vol]);
					
					// Q- and Lambda2-criteria.
					tmp_data[kap+11*vol] = 
						vel_grad[0+3*0]*vel_grad[1+3*1] + vel_grad[1+3*1]*vel_grad[2+3*2] + vel_grad[2+3*2]*vel_grad[0+3*0] +
						-vel_grad[0+3*1]*vel_grad[1+3*0] - vel_grad[1+3*2]*vel_grad[2+3*1] - vel_grad[2+3*0]*vel_grad[0+3*2]
					;
					Eigen::Matrix3f m_S = 0.5f*(m_velgrad + m_velgrad.transpose());
					Eigen::Matrix3f m_O = 0.5f*(m_velgrad - m_velgrad.transpose());
					Eigen::Matrix3f m_A = m_S*m_S + m_O*m_O;
					Eigen::Vector3cf eigvals = m_A.eigenvalues();
					std::sort(eigvals.begin(), eigvals.end(), complex_descending());
					tmp_data[kap+12*vol] = (double)(std::real(eigvals(1)));
				}
			}
		}
		std::cout << "    Finished computing properties..." << std::endl;
		std::cout << std::endl;
		
		
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
		std::cout << "[-] Inserting data in VTK pointers..." << std::endl;
		//#pragma omp parallel for
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
				tmp_data[kap+ 11*vol]
			);
		}
		std::cout << "    Finished inserting data in VTK pointers..." << std::endl;
		
		
		// Printing.
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
		//#pragma omp parallel for
		for (long int kap = 0; kap < vol; kap++)
		{
			if (tmp_data[kap + 9*vol] < 0)
				ghosts->SetValue(kap, ghosts->GetValue(kap) | vtkDataSetAttributes::HIDDENCELL);
		}
		std::cout << "    Finished creating uniform grid..." << std::endl;
		//|
		std::cout << "Finished building VTK dataset, writing..." << std::endl;
		std::string file_name = dirname + std::string("out_") + std::to_string(Kt) + ".vti";
		vtkNew<vtkXMLImageDataWriter> writer;
		//writer->SetInputData(cell_to_points->GetImageDataOutput());
		writer->SetInputData(grid);
		writer->SetFileName(file_name.c_str());
		writer->Write();
		std::cout << "Finished writing VTK dataset..." << std::endl;
		
		
		// Rendering.
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
				// Contour for vorticity magnitude.
			cell_to_points->SetInputData(grid);
			cell_to_points->Update();
			cell_to_points->GetImageDataOutput()->GetPointData()->SetActiveScalars("Vorticity Magnitude");
			contour->SetInputConnection(0, cell_to_points->GetOutputPort(0));
			contour->SetNumberOfContours(3);
			contour->SetValue(0, 0.15);
			contour->SetValue(1, 0.2);
			contour->SetValue(2, 0.3);
			std::cout << "    Finished creating contours..." << std::endl;
			
			std::cout << "[-] Setting up renderer..." << std::endl;
				// Setup offscreen rendering.
			//vtkNew<vtkNamedColors> colors;
			//vtkNew<vtkGraphicsFactory> graphics_factory;
			graphics_factory->SetOffScreenOnlyMode(1);
			graphics_factory->SetUseMesaClasses(1);
				// Create mapper.
			//vtkNew<vtkPolyDataMapper> pd_mapper;
			pd_mapper->SetInputConnection(contour->GetOutputPort(0));
				// Create actor.
			//vtkNew<vtkActor> actor;
			actor->SetMapper(pd_mapper);
			actor->GetProperty()->SetColor(colors->GetColor3d("White").GetData());
				// Create renderer.
			//vtkNew<vtkRenderer> renderer;
			//vtkNew<vtkRenderWindow> renderWindow;
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
		std::cout << "    Finished setup, rendering..." << std::endl;
		renderWindow->SetSize(2048, 2048);
		renderWindow->Render();
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
		std::cout << "    Finished taking photo (no. " << Kt << ")..." << std::endl;
		
		
		// Free allocations.
		delete[] tmp_data;
		delete[] tmp_data_b;
		
		}
		}
	}
	
	
	// Free allocations.
	delete[] mult_f;
	delete[] buffer;
	
	return 0;
}
