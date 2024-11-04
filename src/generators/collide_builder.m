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
fileID_d2q9 = fopen("../solver_lbm_collide_d2q9.cu",'w');
fileID_d3q19 = fopen("../solver_lbm_collide_d3q19.cu",'w');
fileID_d3q27 = fopen("../solver_lbm_collide_d3q27.cu",'w');
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
	args_2 = {"int *cblock_ID_onb", "int *cblock_ID_nbr", "int *cblock_ID_nbr_child", "int *cblock_ID_mask", "int n_maxcblocks"};
	args_3 = {"int *cells_ID_mask", "ufloat_t *cells_f_F"};
	args = {args_1, args_2, args_3};
	build_kernel_header(fileID{K}, 0, sprintf("Collide_%s", l_dq_names{K}), args);
	n_ind = add_bracket(fileID{K}, n_ind, 1, true);

	
	% Declare variables.
	%
	% - Main
	%      + s_ID_cblock:     Shared memory array for block Ids.
	%
	add_variable(fileID{K}, n_ind, "__shared__ int", "s_ID_cblock[M_CBLOCK]");
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
		n_ind = add_condition(fileID{K}, n_ind, " i_kap_b > -1 && (i_kap_bc < 0 || block_on_boundary == 1) ", false);
	%	n_ind = add_for(fileID{K}, n_ind, "l = 0; l < M_CQUAD; l++", false);
		
	
	
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

	
	% Process LES model and get updated relaxation rates (unused).
	LES_builder(fileID{K}, n_ind, dims(K), l_dqs(K), c_all{K}, w_all{K}, 1);
	
	
	
	% Compute post-collision DDF values.
	COMMENT(fileID{K}, n_ind, "Collision step.");
	collide_BGK(fileID{K}, n_ind, dims(K), l_dqs(K), c_all{K}, w_all{K});
	add_line(fileID{K});
	
	
	% Impose boundary conditions.
	% - Read BC input files with 'get_bc'.
	% - Check if on boundary, then implement BC.
	%      + For each neighbor P, check whether DDF direction Q is involved in BC imposition.
	%      + Use 'get_minimal_BC_cond' to construct condition and 'BC_string' to build the expression.
	COMMENT(fileID{K}, n_ind, "Impose boundary conditions.");
	H = get_BC(dims(K));
	[rH, rC] = size(H);
	%
	add_statement(fileID{K}, n_ind, "block_on_boundary = cblock_ID_onb[i_kap_b]", true);
	n_ind = add_condition(fileID{K}, n_ind, "block_on_boundary == 1", false);
	for P = 2:l_dqs(K)
		COMMENT(fileID{K}, n_ind, sprintf("nbr %i", P-1));
		add_statement(fileID{K}, n_ind, sprintf("nbr_kap_b = cblock_ID_nbr[i_kap_b + %i*n_maxcblocks]", P-1), true);
	
		for Q = 2:l_dqs(K)
			QinP = true;
			c_P = c_all{K}(:,P);
			c_Q = c_all{K}(:,Q);
			if (c_P(1)~=c_Q(1) && c_P(1)~=0)
				QinP = false;
			end
			if (c_P(2)~=c_Q(2) && c_P(2)~=0)
				QinP = false;
			end
			if (dims(K) == 3)
				if (c_P(3)~=c_Q(3) && c_P(3)~=0)
					QinP = false;
				end
			end
		
			if (QinP)
				COMMENT(fileID{K}, n_ind+1, sprintf("p = %i", Q-1));
				s_minimal_cond = get_minimal_BC_cond(dims(K), l_dqs(K), Q, P, c_all{K});
				n_ind = add_condition(fileID{K}, n_ind, sprintf("%s", s_minimal_cond), false);
			
				for row = 1:rH
					[s_BC_cond, s_BC_cdotu, s_BC_exp] = get_BC_string(dims(K), H(row,:), Q, c_all{K}, w_all{K});
					if (~strcmp(s_BC_cond,""))
						n_ind = add_condition(fileID{K}, n_ind, s_BC_cond, false);
							add_statement(fileID{K}, n_ind, s_BC_cdotu, true);
							add_statement(fileID{K}, n_ind, s_BC_exp, true);
						n_ind = add_condition(fileID{K}, n_ind, "", true);
					end
				end
			
				n_ind = add_condition(fileID{K}, n_ind, "", true);
			end
		end
	end
	n_ind = add_condition(fileID{K}, n_ind, "", true);
	%
	add_line(fileID{K});
	
	
	% Write updated DDFs to global memory.
	COMMENT(fileID{K}, n_ind, "Write fi* to global memory.");
	store_f(fileID{K}, n_ind, dims(K), l_dqs(K), pb_all{K});
	add_statement(fileID{K}, n_ind, "__syncthreads()", true);
	
	
	% Close bracket of 'Loop over blocks in shared memory'.
	%n_ind = add_for(fileID{K}, n_ind, "", true);
	n_ind = add_for(fileID{K}, n_ind, "", true);
	n_ind = add_condition(fileID{K}, n_ind, "", true);
	
	% Close bracket of kernel header.
	n_ind = add_bracket(fileID{K}, n_ind, 1, false);
	
	
	% Build C++ routine to call collision kernel.
	add_line(fileID{K});
	add_line(fileID{K});
	%
	args_routine = {"int i_dev", "int L"};
	args_1 = {"mesh->n_ids[i_dev][L]", "&mesh->c_id_set[i_dev][L*n_maxcblocks]", "mesh->n_maxcells", "mesh->dxf_vec[L]", "tau_vec[L]", "tau_ratio_vec_C2F[L]", "v0"};
	args_2 = {"mesh->c_cblock_ID_onb[i_dev]", "mesh->c_cblock_ID_nbr[i_dev]", "mesh->c_cblock_ID_nbr_child[i_dev]", "mesh->c_cblock_ID_mask[i_dev]", "mesh->n_maxcblocks"};
	args_3 = {"mesh->c_cells_ID_mask[i_dev]", "mesh->c_cells_f_F[i_dev]"};
	args_kernel = {args_1, args_2, args_3};
	variation_1 = ...
	{...
		'condition', "mesh->n_ids[i_dev][L] > 0",...
		'params', "(M_CBLOCK+mesh->n_ids[i_dev][L]-1)/M_CBLOCK,M_CBLOCK,0,mesh->streams[i_dev]",...
		'args', args_kernel,...
		'template', ""...
	};
	variations = {variation_1};
	%
	build_kernel_routine(fileID{K}, 0, "Solver_LBM::S", sprintf("Collide_%s", l_dq_names{K}), args_routine, variations);
	add_line(fileID{K});

	
	% Close file header.
	add_statement(fileID{K}, 0, "#endif", false);
end

% Close files.
r = zeros(K,1);
for K = 1:length(l_dqs)
	fclose(fileID{K});
end
if (r==0)
	printf("Collision code: Done.\n")
else
	printf("Collision code: Error.\n")
end
