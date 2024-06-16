/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#include "main.h"



int main(int argc, char *argv[])
{
	//
	  // ===================
	  // === Parameters: ===
	  // ===================
	//
	
	double		Lx		= L_c;
	double		Ly		= L_fy*L_c;
	double		Lz		= L_fz*L_c;
	int		Nx_init		= Nx;
	int		Ny_init		= (int)(Nx*(Ly/Lx));
	int		Nz_init		= (int)(Nx*(Lz/Lx));	if (N_DIM == 2) Nz_init = 1;
	int		Nxi[3]		= {Nx_init, Ny_init, Nz_init};
	double		dx		= Lx/Nx;
#if (N_PROBE_AVE==1)
	int		N_iters_ave	= 0;
#endif
	std::ofstream   iter_printer;
	std::ofstream   ref_printer;
	std::ofstream   adv_printer;
	std::ofstream	force_printer;
	
	
	
	//
	  // ==============================
	  // === Create solver objects: ===
	  // ==============================
	//
	
	// MESH
	tic("[Constructing mesh object]:");
	Mesh mesh(Nxi, dx);
	toc("[Constructing mesh object]");
	
	// SOLVER
	tic("[Constructing solver object]:");
	Solver_LBM solver(&mesh);
	toc("[Constructing solver object]");
	
	
	
	//
	  // =======================
	  // === Pre-processing: ===
	  // =======================
	//
	
	// Set initial conditions on the coarse grid.
	solver.S_Initialize(0,0);
	
	// Prepare statistics counters.
	iter_printer.open(std::string(P_DIR_NAME) + "iter_counter.txt");
	adv_printer.open(std::string(P_DIR_NAME) + "time_counter.txt");
	ref_printer.open(std::string(P_DIR_NAME) + "refine_counter.txt");
	force_printer.open(std::string(P_DIR_NAME) + "forces.txt");
		// ref
	ref_printer << "iter mAve mIntComp mRed iter nR nC Pre S1 S2 S3 nCu S4 S5 S6 S7 S8 total\n";
		// adv
	adv_printer << "iter ";
	for (int L = 0; L < MAX_LEVELS; L++) adv_printer << "n" << L << " ";
	for (int L = 0; L < MAX_LEVELS; L++) adv_printer << L << "-Interp " << L << "-Collide " << L << "-Stream " << L << "-Average ";
	adv_printer << "MLUPS" << std::endl;
	
	
	
#if (MAX_LEVELS>1 && (MAX_LEVELS!=N_LEVEL_START+1))
	// Refine near walls.
	for (int L = 0; L < N_LEVEL_START; L++)
	{
		std::cout << "Refining to get to starting level [L=" << L << "]..." << std::endl;
		
		// Mark cells near the wall within certain distance.
		solver.S_ComputeRefCriteria(0,L,2);
		cudaDeviceSynchronize();
		
		// Refine.
		solver.S_UpdateMesh(0, &ref_printer);
		cudaDeviceSynchronize();
	}
	solver.S_Initialize(0,N_LEVEL_START);
	
	for (int L = N_LEVEL_START; L < (MAX_LEVELS)-1; L++)
	{
		std::cout << "Near wall refinement #" << L+1 << std::endl;
		
		// Mark cells near the wall within certain distance.
		solver.S_ComputeRefCriteria(0,L,0);
		cudaDeviceSynchronize();
		cudaDeviceSynchronize();
		
		// Refine.
		solver.S_UpdateMesh(0, &ref_printer);
		cudaDeviceSynchronize();
		cudaDeviceSynchronize();
		
		// Initialize data on higher grid levels.
		solver.S_Initialize(0,L+1);
	}
	
	// Freeze mesh: these new near-wall cells are not eligible for coarsening.
	solver.S_FreezeMesh(0);
#endif
#if (MAX_LEVELS>1 && (MAX_LEVELS==N_LEVEL_START+1))
	for (int L = 0; L < MAX_LEVELS-1; L++)
	{
		std::cout << "Refining to get to starting level [L=" << L << "]..." << std::endl;
		
		// Mark cells near the wall within certain distance.
		solver.S_ComputeRefCriteria(0,L,2);
		cudaDeviceSynchronize();
		
		// Refine.
		solver.S_UpdateMesh(0, &ref_printer);
		cudaDeviceSynchronize();
	}
	solver.S_Initialize(0,N_LEVEL_START);
#endif
	
	
	
	//
	  // ====================
	  // === Solver loop: ===
	  // ====================
	//
	
	// Primary loop. Total of N_PRINT*P_PRINT iterations (number of prints x number of iterations per print).
	for (int i = 0; i < N_PRINT*P_PRINT + 1; i++)
	//for (int i = 0; i < 1; i++)
	{				
		// Print iteration.
		std::cout << "Iteration " << i << ", t = " << i*mesh.dxf_vec[N_LEVEL_START] << std::endl;
		iter_printer << "Iteration " << i << ", t = " << i*dx << " | ";
		for (int L = 0; L < MAX_LEVELS; L++)
			iter_printer << "N" << L << "=" << mesh.n_ids[0][L] << ", ";
		iter_printer << std::endl;
		
		
		
		// Reset advancement time counters, then output the grid hierarchy sizes for computation later.
		double tmp_arr[4*MAX_LEVELS]; for (int L = 0; L < 4*MAX_LEVELS; L++) tmp_arr[L] = 0.0;
		adv_printer << i << " ";
		for (int L = 0; L < MAX_LEVELS; L++)
			adv_printer << mesh.n_ids[0][L] << " ";
		
		// Advance w.r.t the coarse grid.
		solver.S_Advance(0,N_LEVEL_START, &adv_printer, tmp_arr);
		cudaDeviceSynchronize();
		
		
		
		// Refinement stage, performed every P_REFINE iterations.
#if (MAX_LEVELS>1 && (MAX_LEVELS!=N_LEVEL_START+1))
		if (i%P_REFINE == 0 && i > 0)
		{
#if (N_PROBE_AVE==1)
			if (i <= N_PROBE_AVE_START)
			{
#endif
				// Output to refinement time counter (includes grid hierarchy sizes of last iteration).
				std::cout << "Refining... " << i << ", t = " << i*dx << std::endl;
				std::cout << "(last) "; for (int L = 0; L < MAX_LEVELS; L++) std::cout << mesh.n_ids[0][L] << " "; std::cout << std::endl;
				
				// Global average so that data is safely interpolated to new cells.
				std::cout << "Grid average..." << std::endl;
				tic_simple("");
				// |
				for (int L = MAX_LEVELS-1; L >= N_LEVEL_START; L--)
					solver.S_Average(0,L,V_AVERAGE_GRID); // [TODO]
				// |
				cudaDeviceSynchronize();
				ref_printer << i << " " << toc_simple("",T_US) << " ";
				
				// Interpolate to ghost cells so refinement criterion is not spoiled.
				std::cout << "Interpolating to ghost cells..." << std::endl;
				tic_simple("");
				// |
				for (int L = N_LEVEL_START; L < MAX_LEVELS; L++)
				{	
					if (L < MAX_LEVELS-1)
						solver.S_Interpolate(0,L,V_INTERP_INTERFACE);
				}
				// |
				cudaDeviceSynchronize();
				ref_printer << toc_simple("",T_US) << " ";
				
				// Compute the refinement criterion on all cells.
				std::cout << "Reducing..." << std::endl;
				tic_simple("");
				// |
				solver.S_ComputeRefCriteria(0,0,1);
				// |
				cudaDeviceSynchronize();
				ref_printer << toc_simple("",T_US) << " ";
				
				// Refine.
				solver.S_UpdateMesh(i, &ref_printer);
				cudaDeviceSynchronize();
				
#if (N_PROBE_AVE==1)
			}
#endif
		}
#endif
		
		
		
		// Convergence check, performed every N_PROBE_FREQ iterations.
#if (N_PROBE==1)
		if (i%N_PROBE_FREQUENCY == 0 && i > 0)
		{
			ufloat_t ctol = N_Pf(0.0);
		
			mesh.M_RetrieveFromGPU();
			std::cout << "Checking convergence..." << std::endl;
			ctol = mesh.M_CheckConvergence(0);
			if (ctol < V_PROBE_TOL)
			{
				std::cout << "Converged..." << std::endl;
				i = N_PRINT*P_PRINT;
			}
			else
				std::cout << "Not converged...(tol = " << ctol << ")" << std::endl;
		}
#endif
		
		
		
#if (N_PROBE_AVE==1)
		if (i%N_PROBE_AVE_FREQUENCY == 0 && i > N_PROBE_AVE_START)
		{
			std::cout << "Updating averages..." << std::endl;
			// Ensure data is valid on all cells. Global average, then interpolate to ghost cells.
			for (int L = MAX_LEVELS-1; L >= 0; L--)
				solver.S_Average(0,L,V_AVERAGE_GRID);
			for (int L = 0; L < MAX_LEVELS; L++)
				solver.S_Interpolate(0,L,V_INTERP_INTERFACE);
			
			mesh.M_RetrieveFromGPU();
			mesh.M_UpdateMeanVelocities(0, N_iters_ave);
			N_iters_ave++;
		}
#endif
		
		
		
		// Printing stage, performed every P_PRINT iterations.
		if (i%P_PRINT == 0 && i > 0)
		{
			std::cout << "Printing on iteration " << i << " (t = " << i*dx << ")..." << std::endl;
			
			// Ensure data is valid on all cells. Global average, then interpolate to ghost cells.
			for (int L = MAX_LEVELS-1; L >= 0; L--)
				solver.S_Average(0,L,V_AVERAGE_GRID);
			for (int L = 0; L < MAX_LEVELS; L++)
				solver.S_Interpolate(0,L,V_INTERP_INTERFACE);
			
			// Retrieve data from the GPU and print to .vthb file.
			mesh.M_RetrieveFromGPU();
			std::cout << "Writing..." << std::endl;
			mesh.M_Print(0, i);
			std::cout << "Finished printing..." << std::endl;
		}
		
		
		
		// Print lift and drag forces to output if applicable. Temporary measure, will be refined later.
#if (N_DIM==2 && N_PROBE_FORCE==1)
		if (i%N_PROBE_F_FREQUENCY == 0 && i*mesh.dxf_vec[N_LEVEL_START] > 140)
		{
			std::cout << "Printing forces..." << std::endl;
			for (int L = MAX_LEVELS-1; L >= N_LEVEL_START; L--)
				solver.S_Average(0,L,V_AVERAGE_GRID); // [TODO]
			
			mesh.M_RetrieveFromGPU();
			mesh.M_PrintForces(0, std::max(0,MAX_LEVELS_INTERIOR-1), &force_printer);
			//mesh.M_PrintForces(0, std::max(0,num_levels-1), &force_printer);
		}
#endif
	}
	
	// Close all statistics counters.
	iter_printer.close();
	ref_printer.close();
	adv_printer.close();
	force_printer.close();
	
	
	
	return 0;
}
