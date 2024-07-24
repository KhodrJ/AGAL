clc
clear all
close all

warning ("off", "Octave:data-file-in-path");
addpath("./general/");
addpath("./math/");
addpath("./lbm/");
addpath("../../input/");



% Get the velocity sets and their properties.
[c_all, w_all, pb_all] = velocity_sets();
l_dqs = [9,19,27];
dims = [2,3,3];
l_dq_names = {'d2q9', 'd3q19', 'd3q27'};



% Open files.
fileID_d2q9 = fopen("./out/solver_lbm_interp_linear_d2q9.cu",'w');
fileID_d3q19 = fopen("./out/solver_lbm_interp_linear_d3q19.cu",'w');
fileID_d3q27 = fopen("./out/solver_lbm_interp_linear_d3q27.cu",'w');
fileID = {fileID_d2q9, fileID_d3q19, fileID_d3q27};



% For each file, produce the kernel and corresponding C++ routine.
for K = 1:3
	% Initialize indentation tracker.
	n_ind = 0;
	
	% Build file header.
	add_statement(fileID{K}, 0, "#include \"mesh.h\"", false);
	add_statement(fileID{K}, 0, "#include \"solver.h\"", false);
	add_line(fileID{K});
	add_statement(fileID{K}, 0, sprintf("#if (N_Q==%i)", l_dqs(K)), false);
	fprintf(fileID{K}, "\n");


	% Build kernel header.
	args_1 = {"int n_ids_idev_L", "int *id_set_idev_L", "long int n_maxcells", "ufloat_t dx_L", "ufloat_t tau_L", "ufloat_t tau_ratio", "ufloat_t v0"};
	args_2 = {"int *cblock_ID_onb", "int *cblock_ID_mask", "int *cblock_ID_nbr_child", "int n_maxcblocks"};
	args_3 = {"int *cells_ID_mask", "ufloat_t *cells_f_F"};
	args = {args_1, args_2, args_3};
	templates = {"int interp_type = 0"};
	build_kernel_header(fileID{K}, 0, sprintf("Interpolate_Linear_%s", l_dq_names{K}), args, templates);
	n_ind = add_bracket(fileID{K}, n_ind, 1, true);
	
	
	% Declare variables.
	%
	% - Main
	%      + s_ID_cblock:     Shared memory array for block Ids.
	%      + s_F:             Shared memory array for interpolation of f_i's.
	%      + s_Feq:           Shared memory array for interpolation of feq_i's.
	%      + s_tau:           Shared memory array for interpolation of tau.
	%
	add_variable(fileID{K}, n_ind, "__shared__ int", "s_ID_cblock[M_CBLOCK]");
	add_variable(fileID{K}, n_ind, "__shared__ ufloat_t", "s_F[M_CBLOCK]");
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
	%      + omeg:          For computing relaxation rate.
	%      + omegp:         For compute relaxation rate complement.
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
	add_variable(fileID{K}, n_ind, "ufloat_t", "omeg = dx_L / tau_L");
	add_variable(fileID{K}, n_ind, "ufloat_t", "omegp = N_Pf(1.0) - omeg");
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
	add_statement(fileID{K}, n_ind, "// Loop over block Ids.", false);
	n_ind = add_for(fileID{K}, n_ind, "int k = 0; k < M_CBLOCK; k++", false);
		add_statement(fileID{K}, n_ind, "int i_kap_b = s_ID_cblock[k]", true);
		add_statement(fileID{K}, n_ind, "int nbr_kap_b = -1", true);
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
		n_ind = add_condition(fileID{K}, n_ind, " i_kap_b > -1 && ((interp_type == 0 && block_on_boundary == 1) || (interp_type == 1 && cells_ID_mask[i_kap_b] == V_REF_ID_MARK_REFINE)) ", false);
		
		
	% Load DDFs and compute macroscopic properties.
	% - Load DDFs and calculate macroscopic properties.
	% - If using LES or outflow boundary conditions, load velocity into shared memory.
	COMMENT(fileID{K}, n_ind, "Load DDFs and compute macroscopic properties.");
	load_f_alt(fileID{K}, n_ind, dims(K), l_dqs(K), pb_all{K});
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
	LES_builder(fileID{K}, n_ind, dims(K), l_dqs(K), c_all{K}, w_all{K}, 2);
		
		
	% Interpolate to children if applicable.
	COMMENT(fileID{K}, n_ind, "Interpolate rescaled fi to children if applicable.");
	for P = 1:l_dqs(K)
		COMMENT(fileID{K}, n_ind+1, sprintf("p = %i", P-1));
		add_statement(fileID{K}, n_ind, sprintf("cdotu = %s", get_cdotu(c_all{K}(:,P),dims(K))), true);
		add_statement(fileID{K}, n_ind, sprintf("tmp_i = N_Pf(%17.15f)*rho_kap*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)", w_all{K}(P)), true);
		%
		add_statement(fileID{K}, 0, "#if (S_LES==0)", false);
			add_statement(fileID{K}, n_ind, sprintf("s_F[threadIdx.x] = tmp_i + (f_%i - tmp_i)*(tau_ratio)", P-1), true);
		add_statement(fileID{K}, 0, "#else", false);
			add_statement(fileID{K}, n_ind, sprintf("s_F[threadIdx.x] = f_%i", P-1), true);
			add_statement(fileID{K}, n_ind, sprintf("s_Feq[threadIdx.x] = tmp_i", P-1), true);
		add_statement(fileID{K}, 0, "#endif", false);
		add_statement(fileID{K}, n_ind, sprintf("__syncthreads()", P-1), true);
		n_ind = add_bracket(fileID{K}, n_ind, 1, true);
			add_statement(fileID{K}, 0, "#if (S_LES==0)", false);
				vars_s1 = {sprintf("cells_f_F[(i_kap_bc+%%i)*M_CBLOCK + threadIdx.x + %i*n_maxcells]", pb_all{K}(P))};
				vars_s2 = {"s_F"};
				conds = "(interp_type == 0 && cells_ID_mask[(i_kap_bc+%i)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1)";
				interp_linear_tree(fileID{K}, n_ind, dims(K), vars_s1, vars_s2, conds);
			add_statement(fileID{K}, 0, "#else // Storing interpolated fi_eq, fi, tau_ratio in tmp_i, tmp_j and tmp_k, respectively.", false);
				vars_s1 = {"tmp_i", "tmp_j", "tmp_k"};
				vars_s2 = {"s_Feq", "s_F", "s_tau"};
				conds = "(interp_type == 0 && cells_ID_mask[(i_kap_bc+%i)*M_CBLOCK + threadIdx.x]==2) || (interp_type == 1)";
				statements = {
					sprintf("cells_f_F[(i_kap_bc+%%i)*M_CBLOCK + threadIdx.x + %i*n_maxcells] = tmp_i + (tmp_j - tmp_i)*( N_Pf(1.0) - N_Pf(0.25)*dx_L/tmp_k )", pb_all{K}(P))
				};
				interp_linear_tree(fileID{K}, n_ind, dims(K), vars_s1, vars_s2, conds, statements);
			add_statement(fileID{K}, 0, "#endif", false);
			add_statement(fileID{K}, n_ind, "__syncthreads()", true);
		n_ind = add_bracket(fileID{K}, n_ind, 1, false);
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
	args_routine = {"int i_dev", "int L", "int var", "ufloat_t tau_L", "ufloat_t tau_ratio_L"};
	args_1_1 = {"mesh->n_ids[i_dev][L]", "&mesh->c_id_set[i_dev][L*n_maxcblocks]", "n_maxcells", "mesh->dxf_vec[L]", "tau_L", "tau_ratio_L", "v0"};
	args_1_2 = {"mesh->c_cblock_ID_onb[i_dev]", "mesh->c_cblock_ID_mask[i_dev]", "mesh->c_cblock_ID_nbr_child[i_dev]", "n_maxcblocks"};
	args_1_3 = {"mesh->c_cells_ID_mask[i_dev]", "mesh->c_cells_f_F[i_dev]"};
	args_kernel_1 = {args_1_1, args_1_2, args_1_3};
	variation_1 = ...
	{...
		'condition', "mesh->n_ids[i_dev][L] > 0 && var == V_INTERP_INTERFACE",...
		'params', "(M_CBLOCK+mesh->n_ids[i_dev][L]-1)/M_CBLOCK,M_CBLOCK,0,mesh->streams[i_dev]",...
		'args', args_kernel_1,...
		'template', "<0>"...
	};
	args_2_1 = {"mesh->n_ids[i_dev][L]", "&mesh->c_id_set[i_dev][L*n_maxcblocks]", "n_maxcells", "mesh->dxf_vec[L]", "tau_L", "tau_ratio_L", "v0"};
	args_2_2 = {"mesh->c_cblock_ID_onb[i_dev]", "mesh->c_cblock_ID_mask[i_dev]", "mesh->c_cblock_ID_nbr_child[i_dev]", "n_maxcblocks"};
	args_2_3 = {"mesh->c_cblock_ID_ref[i_dev]", "mesh->c_cells_f_F[i_dev]"};
	args_kernel_2 = {args_2_1, args_2_2, args_2_3};
	variation_2 = ...
	{...
		'condition', "mesh->n_ids[i_dev][L] > 0 && var == V_INTERP_ADDED",...
		'params', "(M_CBLOCK+mesh->n_ids[i_dev][L]-1)/M_CBLOCK,M_CBLOCK,0,mesh->streams[i_dev]",...
		'args', args_kernel_2,...
		'template', "<1>"...
	};
	variations = {variation_1, variation_2};
	%
	build_kernel_routine(fileID{K}, 0, "Solver_LBM::S", sprintf("Interpolate_Linear_%s", l_dq_names{K}), args_routine, variations);
	add_line(fileID{K});

	
	% Close file header.
	add_statement(fileID{K}, 0, "#endif", false);
end

% Close files.
for K = 1:length(l_dqs)
	fclose(fileID{K});
end

% Copy to solver directory.
r = system("cp ./out/solver_lbm_interp_linear_* ../");
if (r==0)
	printf("Interpolation code: Done.\n")
else
	printf("Interpolation code: Error.\n")
end
