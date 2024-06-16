clc
clear all
close all

warning ("off", "Octave:data-file-in-path");
addpath("./general/");
addpath("./math/");
addpath("./lbm/");
addpath("./input/");



% Get the velocity sets and their properties.
[c_all, w_all, pb_all] = velocity_sets();
l_dqs = [9,19,27];
dims = [2,3,3];
l_dq_names = {'d2q9', 'd3q19', 'd3q27'};



% Open files.
fileID_d2q9 = fopen("./out/mesh_comm_average_d2q9.cu",'w');
fileID_d3q19 = fopen("./out/mesh_comm_average_d3q19.cu",'w');
fileID_d3q27 = fopen("./out/mesh_comm_average_d3q27.cu",'w');
fileID = {fileID_d2q9, fileID_d3q19, fileID_d3q27};



% For each file, produce the kernel and corresponding C++ routine.
for K = 1:3
	% Initialize indentation tracker.
	n_ind = 0;
	
	% Build file header.
	add_statement(fileID{K}, 0, "#include \"mesh.h\"", false);
	add_line(fileID{K});
	add_statement(fileID{K}, 0, sprintf("#if (N_Q==%i)", l_dqs(K)), false);
	fprintf(fileID{K}, "\n");


	% Build kernel header.
	args_1 = {"int n_ids_idev_L", "int *id_set_idev_L", "long int n_maxcells", "ufloat_t dx_Lm1", "ufloat_t dx_L", "ufloat_t tau_L", "ufloat_t tau_ratio"};
	args_2 = {"int *cblock_ID_onb", "int *cblock_ID_mask", "int *cblock_ID_nbr_child", "int n_maxcblocks"};
	args_3 = {"int *cells_ID_mask", "ufloat_t *cells_f_F"};
	args = {args_1, args_2, args_3};
	templates = {"int ave_type = 0"};
	build_kernel_header(fileID{K}, 0, sprintf("Average_%s", l_dq_names{K}), args, templates);
	n_ind = add_bracket(fileID{K}, n_ind, 1, true);
	
	
	% Declare variables.
	%
	% - Main
	%      + s_ID_cblock:     Shared memory array for block Ids.
	%      + s_ID_mask_child: Shared memory array for cell masks.
	%      + s_Fc:            Shared memory array for averaging of f_i's.
	%      + s_Feq:           Shared memory array for averaging of feq_i's.
	%      + s_tau:           Shared memory array for averaging of tau.
	%
	add_variable(fileID{K}, n_ind, "__shared__ int", "s_ID_cblock[M_CBLOCK]");
	add_variable(fileID{K}, n_ind, "__shared__ int", "s_ID_mask_child[M_CBLOCK]");
	add_variable(fileID{K}, n_ind, "__shared__ ufloat_t", "s_Fc[M_CBLOCK]");
	add_statement(fileID{K}, 0, "#if (S_LES==1)", false);
		add_variable(fileID{K}, n_ind, "__shared__ ufloat_t", "s_Feq[M_CBLOCK]");
		add_variable(fileID{K}, n_ind, "__shared__ ufloat_t", "s_tau[M_CBLOCK]");
	add_statement(fileID{K}, 0, "#endif", false);
	%
	% - Open boundaries
	%      + s_{u/v/w}:       Shared memory array for {x/y/z}- velocity component.
	%
	add_statement(fileID{K}, 0, "#if (B_TYPE==1||S_LES==1)", false);
	if (K == 1)
		add_variable(fileID{K}, n_ind, "__shared__ ufloat_t", "s_u[(Nbx+2)*(Nbx+2)]");
		add_variable(fileID{K}, n_ind, "__shared__ ufloat_t", "s_v[(Nbx+2)*(Nbx+2)]");
	else
		add_variable(fileID{K}, n_ind, "__shared__ ufloat_t", "s_u[(Nbx+2)*(Nbx+2)*(Nbx+2)]");
		add_variable(fileID{K}, n_ind, "__shared__ ufloat_t", "s_v[(Nbx+2)*(Nbx+2)*(Nbx+2)]");
		add_variable(fileID{K}, n_ind, "__shared__ ufloat_t", "s_w[(Nbx+2)*(Nbx+2)*(Nbx+2)]");
	end
	add_statement(fileID{K}, 0, "#endif", false);
	%
	% - Id variables
	%      + kap:                Define global index based on block Id, block dim. and thread Id.
	%      + block_on_boundary:  Indicates if block lies on a boundary.
	%      + i_kap:              Stores current block's Id.
	%      + i_kap_bc:           Stores current block's first child Id.
	%      + {I/J/K}_kap:        Local {X/Y/Z}-index coordinate.
	%      + child0_IJK:         Intermediate value for finding the correct child with local index 0.
	%
	add_variable(fileID{K}, n_ind, "int", "kap = blockIdx.x*blockDim.x + threadIdx.x");
	add_variable(fileID{K}, n_ind, "int", "block_on_boundary = 0");
	add_variable(fileID{K}, n_ind, "int", "i_kap = -1");
	add_variable(fileID{K}, n_ind, "int", "i_kap_bc = -1");
	add_variable(fileID{K}, n_ind, "int", "I_kap = threadIdx.x % Nbx");
	add_variable(fileID{K}, n_ind, "int", "J_kap = (threadIdx.x / Nbx) % Nbx");
	if (K > 1)
		add_variable(fileID{K}, n_ind, "int", "K_kap = (threadIdx.x / Nbx) / Nbx");
	end
	if (K == 1)
		add_variable(fileID{K}, n_ind, "int", "child0_IJK = 2*((threadIdx.x % Nbx)%2) + Nbx*(2*(((threadIdx.x / Nbx) % Nbx)%2))");
	else
		add_variable(fileID{K}, n_ind, "int", "child0_IJK = 2*((threadIdx.x % Nbx)%2) + Nbx*(2*(((threadIdx.x / Nbx) % Nbx)%2)) + Nbx*Nbx*(2*(((threadIdx.x / Nbx) / Nbx)%2))");
	end
	%
	% - LBM variables
	%      + f_i:           Density distribution functions.
	%      + rho_kap:       Density at current thread.
	%      + {u/v/w}_kap:   {u/v/w}-component of velocity at current thread.
	%
	%      + tmp_{ij}:      Temporary slot for computing f_eq, to be used in getting t_eff for turbulence modeling.
	%      + tmp_k:         Temporary slot for computing f_eq, to be used in getting t_eff for turbulence modeling.
	%      + tmp_k:         For computing stress-tensor norm.
	%
	%      + cdotu:         For computing c_i dot u.
	%      + udotu:         For computing u dot u.
	%
	for P = 1:l_dqs(K)
		add_variable(fileID{K}, n_ind, "ufloat_t", sprintf("f_%i = N_Pf(0.0)", P-1));
	end
	add_variable(fileID{K}, n_ind, "ufloat_t", "rho_kap = N_Pf(0.0)");
	add_variable(fileID{K}, n_ind, "ufloat_t", "u_kap = N_Pf(0.0)");
	add_variable(fileID{K}, n_ind, "ufloat_t", "v_kap = N_Pf(0.0)");
	if (K > 1)
		add_variable(fileID{K}, n_ind, "ufloat_t", "w_kap = N_Pf(0.0)");
	end
	%
	add_variable(fileID{K}, n_ind, "ufloat_t", "tmp_i = N_Pf(0.0)");
	add_statement(fileID{K}, 0, "#if (S_LES==1)", false);
		add_variable(fileID{K}, n_ind, "ufloat_t", "tmp_j = N_Pf(0.0)");
		add_variable(fileID{K}, n_ind, "ufloat_t", "tmp_k = N_Pf(0.0)");
	add_statement(fileID{K}, 0, "#endif", false);
	%
	add_variable(fileID{K}, n_ind, "ufloat_t", "cdotu = N_Pf(0.0)");
	add_variable(fileID{K}, n_ind, "ufloat_t", "udotu = N_Pf(0.0)");
	add_line(fileID{K});
	
	
	% Read block Ids into shared memory, then loop.
	% - Start reading block Ids from Id set.
	% - Limit thread access by array size.
	% - Load from Id set.
	% - Write to shared memory.
	add_statement(fileID{K}, n_ind, "s_ID_cblock[threadIdx.x] = -1", true);
	n_ind = add_condition(fileID{K}, n_ind, "kap < n_ids_idev_L", false);
		add_statement(fileID{K}, n_ind, "i_kap = id_set_idev_L[kap]", true);
		add_statement(fileID{K}, n_ind, "s_ID_cblock[threadIdx.x] = i_kap", true);
	n_ind = add_condition(fileID{K}, n_ind, "", true);
	add_statement(fileID{K}, n_ind, "__syncthreads()", true);
	add_line(fileID{K});
	
	
	% Loop over blocks in shared memory.
	% - Prepare loop with resets.
	%      + Load current block Ids into register memory.
	%      + Temporary slot for neighbor block Ids.
	%      + Reset first-child Id.
	%      + Reset first-child Id.
	% - Only advance if leaf, not branch.
	%      + Load first child for current block.
	%      + Load block mask to determine interpolation situation.
	% - If conditions met, proceed with collision.
	% - Open up for-loops over children.
	add_statement(fileID{K}, n_ind, "// Loop over block Ids.", false);
	n_ind = add_for(fileID{K}, n_ind, "int k = 0; k < M_CBLOCK; k++", false);
		add_statement(fileID{K}, n_ind, "int i_kap_b = s_ID_cblock[k]", true);
		add_statement(fileID{K}, n_ind, "i_kap_bc = -1", true);
		add_statement(fileID{K}, n_ind, "block_on_boundary = 0", true);
		add_line(fileID{K});
	%
		n_ind = add_condition(fileID{K}, n_ind, "i_kap_b > -1", false);
			add_statement(fileID{K}, n_ind, "i_kap_bc = cblock_ID_nbr_child[i_kap_b]", true);
			add_statement(fileID{K}, n_ind, "block_on_boundary = cblock_ID_mask[i_kap_b]", true);
		n_ind = add_condition(fileID{K}, n_ind, "", true);
		add_line(fileID{K});
	%
		n_ind = add_condition(fileID{K}, n_ind, " i_kap_bc > -1 && (ave_type == 2 || block_on_boundary == 1) ", false);
	%
		if (dims(K)==3)
			n_ind = add_for(fileID{K}, n_ind, "int xc_k = 0; xc_k < 2; xc_k++", false);
		end
				n_ind = add_for(fileID{K}, n_ind, "int xc_j = 0; xc_j < 2; xc_j++", false);
					n_ind = add_for(fileID{K}, n_ind, "int xc_i = 0; xc_i < 2; xc_i++", false);
						if (dims(K)==2)
							add_statement(fileID{K}, n_ind, "int xc = xc_i + 2*xc_j", true);
						else
							add_statement(fileID{K}, n_ind, "int xc = xc_i + 2*xc_j + 4*xc_k", true);
						end
						add_line(fileID{K});
		
		
	% Load DDFs and compute macroscopic properties.
	% - Load DDFs and calculate macroscopic properties.
	% - If using LES or outflow boundary conditions, load velocity into shared memory.
	COMMENT(fileID{K}, n_ind, "Load DDFs and compute macroscopic properties.");
	for P = 1:l_dqs(K)
		add_statement(fileID{K}, n_ind, sprintf("f_%i = cells_f_F[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x + %i*n_maxcells]", P-1, pb_all{K}(P)), true);
	end
	load_macro_props(fileID{K}, n_ind, dims(K), l_dqs(K), c_all{K});
	%
	add_statement(fileID{K}, 0, "#if (B_TYPE==1||S_LES==1)", false);
	load_to_shared_halo(fileID{K}, n_ind, dims(K), "s_u", "u_kap");
	load_to_shared_halo(fileID{K}, n_ind, dims(K), "s_v", "v_kap");
	if (dims(K) == 3)
		load_to_shared_halo(fileID{K}, n_ind, dims(K), "s_w", "w_kap");
	end
	add_statement(fileID{K}, n_ind, "__syncthreads()", true);
	add_statement(fileID{K}, 0, "#endif", false);
	add_line(fileID{K});
		
		
	% LES implementation (unused).
	% LES_builder(fileID{K}, n_ind, dims(K), l_dqs(K), c_all{K}, w_all{K}, 3);
		
		
	% Average to parent if applicable.
	% - Place rescaled DDFs in shared memory one by one and average in parents.
	%      + Get cell masks for child. Only average if the child cells considered are marked for it (i.e. equal to 1).
	%      + If ave_type is 1, average to whole parent block, so masks for all children are set to 1.
	COMMENT(fileID{K}, n_ind, "Average rescaled fi to parent if applicable.");
	add_statement(fileID{K}, n_ind, "s_ID_mask_child[threadIdx.x] = cells_ID_mask[(i_kap_bc + xc)*M_CBLOCK + threadIdx.x]", true);
	n_ind = add_condition(fileID{K}, n_ind, "ave_type > 0 && s_ID_mask_child[threadIdx.x] < 2", false);
		add_statement(fileID{K}, n_ind, "s_ID_mask_child[threadIdx.x] = 1", true);
	n_ind = add_condition(fileID{K}, n_ind, "", true);
	%
	for P = 1:l_dqs(K)
		add_line(fileID{K});
		COMMENT(fileID{K}, n_ind+1, sprintf("p = %i", P-1));
		add_statement(fileID{K}, n_ind, sprintf("cdotu = %s", get_cdotu(c_all{K}(:,P),dims(K))), true);
		add_statement(fileID{K}, n_ind, sprintf("tmp_i = N_Pf(%17.15f)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)", w_all{K}(P)), true);
		%
		add_statement(fileID{K}, 0, "#if (S_LES==0)", false);
			add_statement(fileID{K}, n_ind, sprintf("s_Fc[threadIdx.x] = tmp_i + (f_%i - tmp_i)*tau_ratio", P-1), true);
		add_statement(fileID{K}, 0, "#else", false);
			add_statement(fileID{K}, n_ind, sprintf("s_Fc[threadIdx.x] = f_%i", P-1), true);
			add_statement(fileID{K}, n_ind, sprintf("s_Feq[threadIdx.x] = tmp_i", P-1), true);
		add_statement(fileID{K}, 0, "#endif", false);
		add_statement(fileID{K}, n_ind, "__syncthreads()", true);
		%
		if (dims(K)==2)
			add_statement(fileID{K}, 0, "#if (S_LES==0)", false);
				n_ind = add_condition(fileID{K}, n_ind, "s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j", false);
					add_statement(fileID{K}, n_ind, sprintf("cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + %i*n_maxcells] = N_Pf(0.25)*( s_Fc[(child0_IJK + 0 + Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1)] )", pb_all{K}(P)), true);
				n_ind = add_condition(fileID{K}, n_ind, "", true);
			add_statement(fileID{K}, 0, "#else // Storing interpolated fi_eq, fi, tau_ratio in tmp_i, tmp_j and tmp_k, respectively.", false);
				n_ind = add_condition(fileID{K}, n_ind, "s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j", false);
					add_statement(fileID{K}, n_ind, "tmp_i = N_Pf(0.25)*( s_Feq[(child0_IJK + 0 + Nbx*0)] + s_Feq[(child0_IJK + 1 + Nbx*0)] + s_Feq[(child0_IJK + 0 + Nbx*1)] + s_Feq[(child0_IJK + 1 + Nbx*1)] )", true);
					add_statement(fileID{K}, n_ind, "tmp_j = N_Pf(0.25)*( s_Fc[(child0_IJK + 0 + Nbx*0)] + s_Fc[(child0_IJK + 1 + Nbx*0)] + s_Fc[(child0_IJK + 0 + Nbx*1)] + s_Fc[(child0_IJK + 1 + Nbx*1)] )", true);
					add_statement(fileID{K}, n_ind, "tmp_k = N_Pf(0.25)*( s_tau[(child0_IJK + 0 + Nbx*0)] + s_tau[(child0_IJK + 1 + Nbx*0)] + s_tau[(child0_IJK + 0 + Nbx*1)] + s_tau[(child0_IJK + 1 + Nbx*1)] )", true);
					add_statement(fileID{K}, n_ind, sprintf("cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + %i*n_maxcells] = tmp_i + (tmp_j - tmp_i)*( N_Pf(1.0) + N_Pf(0.25)*dx_Lm1/tmp_k )", pb_all{K}(P)), true);
				n_ind = add_condition(fileID{K}, n_ind, "", true);
			add_statement(fileID{K}, 0, "#endif", false);
		else
			add_statement(fileID{K}, 0, "#if (S_LES==0)", false);
				n_ind = add_condition(fileID{K}, n_ind, "s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k", false);
					add_statement(fileID{K}, n_ind, sprintf("cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + %i*n_maxcells] = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] )", pb_all{K}(P)), true);
				n_ind = add_condition(fileID{K}, n_ind, "", true);
			add_statement(fileID{K}, 0, "#else // Storing interpolated fi_eq, fi, tau_ratio in tmp_i, tmp_j and tmp_k, respectively.", false);
				n_ind = add_condition(fileID{K}, n_ind, "s_ID_mask_child[child0_IJK] == 1 && I_kap >= 2*xc_i && I_kap < 2+2*xc_i && J_kap >= 2*xc_j && J_kap < 2+2*xc_j && K_kap >= 2*xc_k && K_kap <= 2+2*xc_k", false);
					add_statement(fileID{K}, n_ind, "tmp_i = N_Pf(0.125)*( s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Feq[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Feq[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] )", true);
					add_statement(fileID{K}, n_ind, "tmp_j = N_Pf(0.125)*( s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_Fc[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_Fc[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] )", true);
					add_statement(fileID{K}, n_ind, "tmp_k = N_Pf(0.125)*( s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*0)] +  s_tau[(child0_IJK + 0 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*0 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 0 + Nbx*1 + Nbx*Nbx*1)] +  s_tau[(child0_IJK + 1 + Nbx*1 + Nbx*Nbx*1)] )", true);
					add_statement(fileID{K}, n_ind, sprintf("cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + %i*n_maxcells] = tmp_i + (tmp_j - tmp_i)*( N_Pf(1.0) + N_Pf(0.25)*dx_Lm1/tmp_k )", pb_all{K}(P)), true);
				n_ind = add_condition(fileID{K}, n_ind, "", true);
			add_statement(fileID{K}, 0, "#endif", false);
		end
		add_statement(fileID{K}, n_ind, "__syncthreads()", true);
	end
	
	
	% Close for-loops over children.
	n_ind = add_for(fileID{K}, n_ind, "", true);
	n_ind = add_for(fileID{K}, n_ind, "", true);
	if (dims(K)==3)
		n_ind = add_for(fileID{K}, n_ind, "", true);
	end
	
	
	% Close bracket of 'Loop over blocks in shared memory'.
	n_ind = add_for(fileID{K}, n_ind, "", true);
	n_ind = add_condition(fileID{K}, n_ind, "", true);
	
	% Close bracket of kernel header.
	n_ind = add_bracket(fileID{K}, n_ind, 1, false);
	
	
	% Build C++ routine to call collision kernel.
	add_line(fileID{K});
	add_line(fileID{K});
	%
	args_routine = {"int i_dev", "int L", "int var", "ufloat_t Cscale", "ufloat_t Cscale2"};
	args_1_1 = {"n_ids[i_dev][L]", "c_id_set[i_dev][L]", "n_maxcells", "dxf_vec[L+1]", "dxf_vec[L]", "Cscale", "Cscale2"};
	args_1_2 = {"c_cblock_ID_onb[i_dev]", "c_cblock_ID_mask[i_dev]", "c_cblock_ID_nbr_child[i_dev]", "n_maxcblocks"};
	args_1_3 = {"c_cells_ID_mask[i_dev]", "c_cells_f_F[i_dev]"};
	args_kernel_1 = {args_1_1, args_1_2, args_1_3};
	variation_1 = ...
	{...
		'condition', "n_ids[i_dev][L] > 0 && var == V_AVERAGE_INTERFACE",...
		'params', "(M_CBLOCK+n_ids[i_dev][L]-1)/M_CBLOCK,M_CBLOCK,0,streams[i_dev]",...
		'args', args_kernel_1,...
		'template', "<0>"...
	};
	args_2_1 = {"n_ids[i_dev][L]", "c_id_set[i_dev][L]", "n_maxcells", "dxf_vec[L+1]", "dxf_vec[L]", "Cscale", "Cscale2"};
	args_2_2 = {"c_cblock_ID_onb[i_dev]", "c_cblock_ID_mask[i_dev]", "c_cblock_ID_nbr_child[i_dev]", "n_maxcblocks"};
	args_2_3 = {"c_cells_ID_mask[i_dev]", "c_cells_f_F[i_dev]"};
	args_kernel_2 = {args_2_1, args_2_2, args_2_3};
	variation_2 = ...
	{...
		'condition', "n_ids[i_dev][L] > 0 && var == V_AVERAGE_BLOCK",...
		'params', "(M_CBLOCK+n_ids[i_dev][L]-1)/M_CBLOCK,M_CBLOCK,0,streams[i_dev]",...
		'args', args_kernel_2,...
		'template', "<1>"...
	};
	args_3_1 = {"n_ids[i_dev][L]", "c_id_set[i_dev][L]", "n_maxcells", "dxf_vec[L+1]", "dxf_vec[L]", "Cscale", "Cscale2"};
	args_3_2 = {"c_cblock_ID_onb[i_dev]", "c_cblock_ID_mask[i_dev]", "c_cblock_ID_nbr_child[i_dev]", "n_maxcblocks"};
	args_3_3 = {"c_cells_ID_mask[i_dev]", "c_cells_f_F[i_dev]"};
	args_kernel_3 = {args_3_1, args_3_2, args_3_3};
	variation_3 = ...
	{...
		'condition', "n_ids[i_dev][L] > 0 && var == V_AVERAGE_GRID",...
		'params', "(M_CBLOCK+n_ids[i_dev][L]-1)/M_CBLOCK,M_CBLOCK,0,streams[i_dev]",...
		'args', args_kernel_3,...
		'template', "<2>"...
	};
	variations = {variation_1, variation_2, variation_3};
	%
	build_kernel_routine(fileID{K}, 0, "Mesh::M", sprintf("Average_%s", l_dq_names{K}), args_routine, variations);
	add_line(fileID{K});
	
	
	% Close file header.
	add_statement(fileID{K}, 0, "#endif", false);
end

% Close files.
for K = 1:length(l_dqs)
	fclose(fileID{K});
end

% Copy to solver directory.
r = system("cp ./out/mesh_comm_average_* ../mesh/");
if (r==0)
	printf("Averaging code: Done.\n")
else
	printf("Averaging code: Error.\n")
end
