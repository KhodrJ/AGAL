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
	double		Ly		= L_c;
	double		Lz		= L_c;
	int		Nx_init		= Nx;
	int		Ny_init		= Nx*(int)(Ly/Lx);
	int		Nz_init		= Nx*(int)(Lz/Lx);	if (N_DIM == 2) Nz_init = 1;
	int		Nxi[3]		= {Nx_init, Ny_init, Nz_init};
	double		dx		= Lx/Nx;
	std::ofstream   iter_printer;
	std::ofstream   ref_printer;
	std::ofstream   adv_printer;

	//
	  // ==============================
	  // === Create solver objects: ===
	  // ==============================
	//
	
	tic("[Constructing mesh object]:");
	Mesh mesh(Nxi, dx);
	toc("[Constructing mesh object]");
	
	tic("[Constructing solver object]:");
	Solver_LBM solver(&mesh);
	toc("[Constructing solver object]");
	
	
	
	//
	  // =======================
	  // === Pre-processing: ===
	  // =======================
	//
	
	// Set initial conditions.
	solver.S_Initialize(0,0);
	solver.S_ComputeU(0,0);
	
#if (MAX_LEVELS > 1)
	// Refine near walls.
	ref_printer.open(std::string(P_DIR_NAME) + "refine_counter.txt");
	ref_printer << "iter mAve mIntComp mRed nR nC Pre S1 S2 S3 nCu S4 S5 S6 S7 S8\n";
	int num_levels = 2;
	for (int L = 0; L < num_levels; L++)
	{
		std::cout << "Near wall refinement #" << L+1 << std::endl;
		solver.S_ComputeRefCriteria(0,L,0);
		cudaDeviceSynchronize();
		solver.S_UpdateMesh(0, &ref_printer);
		cudaDeviceSynchronize();
		solver.S_Initialize(0,L+1);
		solver.S_ComputeU(0,L+1);
	}
	solver.S_FreezeMesh(0);
#endif
	
	
	
	//
	  // ====================
	  // === Solver loop: ===
	  // ====================
	//
	iter_printer.open(std::string(P_DIR_NAME ) + "iter_counter.txt");
	adv_printer.open(std::string(P_DIR_NAME) + "time_counter.txt");
	adv_printer << "iter ";
	for (int L = 0; L < MAX_LEVELS; L++) adv_printer << "n" << L << " ";
	for (int L = 0; L < MAX_LEVELS; L++) adv_printer << L << "-Interp " << L << "-Collide " << L << "-Stream " << L << "-Average ";
	adv_printer << std::endl;
	for (int i = 0; i < N_PRINT*P_PRINT + 1; i++)
	{				
		// Print iteration.
		std::cout << "Iteration " << i << ", t = " << i*dx << std::endl;
		iter_printer << "Iteration " << i << ", t = " << i*dx << " | ";
		for (int L = 0; L < MAX_LEVELS; L++)
			iter_printer << "N" << L << "=" << mesh.n_ids[0][L] << ", ";
		iter_printer << std::endl;
		
		
		
		//tic_simple("[Advance]");
		double tmp_arr[4*MAX_LEVELS]; for (int L = 0; L < 4*MAX_LEVELS; L++) tmp_arr[L] = 0.0;
		adv_printer << i << " ";
		for (int L = 0; L < MAX_LEVELS; L++)
			adv_printer << mesh.n_ids[0][L] << " ";
			//
		solver.S_Advance(0,0, &adv_printer, tmp_arr);
			//
		cudaDeviceSynchronize();
		//toc_simple("[Advance]",T_US);

		
		
		if (i%P_REFINE == 0 && i > 0)
		{
			std::cout << "Refining... " << i << ", t = " << i*dx << std::endl;
			std::cout << "(last) "; for (int L = 0; L < MAX_LEVELS; L++) std::cout << mesh.n_ids[0][L] << " "; std::cout << std::endl;
			
			tic_simple("");
			for (int L = MAX_LEVELS-1; L >= 0; L--)
			{
				//solver.S_ComputeEq(0,L+1);
				solver.S_Average(0,L,4);
				//solver.S_ComputeEq(0,L);
			}
			cudaDeviceSynchronize();
			ref_printer << i << " " << toc_simple("",T_US) << " ";
			
			tic_simple("");
			for (int L = 0; L < MAX_LEVELS; L++)
			{	
				if (L < MAX_LEVELS-1)
					solver.S_Interpolate(0,L,2);
					
				solver.S_ComputeU(0,L);
				solver.S_ComputeW(0,L);
			}
			cudaDeviceSynchronize();
			ref_printer << toc_simple("",T_US) << " ";
			
			std::cout << "Reducing..." << std::endl;
			tic_simple("");
			solver.S_ComputeRefCriteria(0,0,1);
			cudaDeviceSynchronize();
			ref_printer << toc_simple("",T_US) << " ";

			solver.S_UpdateMesh(i, &ref_printer);
			cudaDeviceSynchronize();
		}
		
		
		
		if (i%P_PRINT == 0 && i > 0)
		{
			std::cout << "Printing on iteration " << i << " (t = " << i*dx << ")..." << std::endl;
			

			for (int L = 0; L < MAX_LEVELS; L++)
			{
				solver.S_Average(0,L,4);
				solver.S_Interpolate(0,L,2);
				
				solver.S_ComputeU(0,L);
				solver.S_ComputeW(0,L);
			}
			
			
			mesh.M_RetrieveFromGPU();
			std::cout << "Writing..." << std::endl;
			mesh.M_Print(0, i);
			std::cout << "Finished printing..." << std::endl;
		}
	}
	iter_printer.close();
	ref_printer.close();
	adv_printer.close();
	
	
	
	return 0;
}
