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
fileID_d2q9 = fopen("../solver_lbm_set_ic_d2q9.cu",'w');
fileID_d3q19 = fopen("../solver_lbm_set_ic_d3q19.cu",'w');
fileID_d3q27 = fopen("../solver_lbm_set_ic_d3q27.cu",'w');
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
	args_1 = {"int n_ids_idev_L", "int *id_set_idev_L", "long int n_maxcells"};
	args_2 = {"ufloat_t *cells_f_F"};
	args_3 = {"ufloat_t rho_t0", "ufloat_t u_t0", "ufloat_t v_t0", "ufloat_t w_t0"};
	args = {args_1, args_2, args_3};
	build_kernel_header(fileID{K}, 0, sprintf("SetInitialConditions_%s", l_dq_names{K}), args);
	n_ind = add_bracket(fileID{K}, n_ind, 1, true);
	
	
	% Declare variables.
	%
	% - Main
	%      + s_ID_cblock:     Shared memory array for block Ids.
	%      + s_F:             Shared memory array for interpolation of f_i's.
	%
	add_variable(fileID{K}, n_ind, "__shared__ int", "s_ID_cblock[M_CBLOCK]");
	%
	% - Id variables
	%      + kap:                Define global index based on block Id, block dim. and thread Id.
	%      + i_kap:              Stores current block's Id.
	%
	add_variable(fileID{K}, n_ind, "int", "kap = blockIdx.x*blockDim.x + threadIdx.x");
	add_variable(fileID{K}, n_ind, "int", "i_kap = -1");
	%
	% - LBM variables
	%      + cdotu:         For computing c_i dot u.
	%      + udotu:         For computing u dot u.
	%
	add_variable(fileID{K}, n_ind, "ufloat_t", "cdotu = N_Pf(0.0)");
	add_variable(fileID{K}, n_ind, "ufloat_t", "udotu = u_t0*u_t0 + v_t0*v_t0 + w_t0*w_t0");
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
		add_line(fileID{K});
	%
		n_ind = add_condition(fileID{K}, n_ind, " i_kap_b > -1 ", false);
	
	
	% Compute post-stream initial DDF values.
	COMMENT(fileID{K}, n_ind, "Compute IC.");
	for P = 1:l_dqs(K)
		add_statement(fileID{K}, n_ind, sprintf("cdotu = %s", get_cdotu_const(c_all{K}(:,P),dims(K))), true);
		add_statement(fileID{K}, n_ind, sprintf("cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + %i*n_maxcells] = N_Pf(%17.15f)*rho_t0*(N_Pf(1.0) + N_Pf(3.0)*cdotu + N_Pf(4.5)*cdotu*cdotu - N_Pf(1.5)*udotu)", pb_all{K}(P), w_all{K}(P)), true);
	end
	add_line(fileID{K});
	
	
	% Close bracket of 'Loop over blocks in shared memory'.
	n_ind = add_for(fileID{K}, n_ind, "", true);
	n_ind = add_condition(fileID{K}, n_ind, "", true);
	
	% Close bracket of kernel header.
	n_ind = add_bracket(fileID{K}, n_ind, 1, false);
	
	
	% Build C++ routine to call IC kernel.
	T = textread("IC.txt");
	add_line(fileID{K});
	add_line(fileID{K});
	%
	args_routine = {"int i_dev", "int L"};
	args_1 = {"mesh->n_ids[i_dev][L]", "&mesh->c_id_set[i_dev][L*n_maxcblocks]", "mesh->n_maxcells"};
	args_2 = {"mesh->c_cells_f_F[i_dev]"};
	args_3 = {sprintf("N_Pf(%17.15f)", T(1)), sprintf("N_Pf(%17.15f)", T(2)), sprintf("N_Pf(%17.15f)", T(3)), sprintf("N_Pf(%17.15f)", T(4))};
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
	build_kernel_routine(fileID{K}, 0, "Solver_LBM::S", sprintf("SetInitialConditions_%s", l_dq_names{K}), args_routine, variations);
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
	printf("Initial condition code: Done.\n")
else
	printf("Initial condition code: Error.\n")
end
