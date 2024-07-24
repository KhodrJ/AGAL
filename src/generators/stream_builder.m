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
[l_dqs_indexed, p_indexed, nbr_quad_indexed] = esoteric_data();
l_dqs = [9,19,27];
dims = [2,3,3];
l_dq_names = {'d2q9', 'd3q19', 'd3q27'};



% Open files.
fileID_d2q9 = fopen("../solver_lbm_stream_inpl_d2q9.cu",'w');
fileID_d3q19 = fopen("../solver_lbm_stream_inpl_d3q19.cu",'w');
fileID_d3q27 = fopen("../solver_lbm_stream_inpl_d3q27.cu",'w');
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
	args_1 = {"int n_ids_idev_L", "int *id_set_idev_L", "long int n_maxcells", "ufloat_t dx_L", "ufloat_t tau_L"};
	args_2 = {"int *cblock_ID_onb", "int *cblock_ID_nbr", "int *cblock_ID_nbr_child", "int *cblock_ID_mask", "int n_maxcblocks"};
	args_3 = {"ufloat_t *cells_f_F"};
	args = {args_1, args_2, args_3};
	build_kernel_header(fileID{K}, 0, sprintf("Stream_Inpl_%s", l_dq_names{K}), args);
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
	if (K == 1)
		add_variable(fileID{K}, n_ind, "__shared__ ufloat_t", "s_F_p[(Nbx+2)*(Nbx+2)]");
		add_variable(fileID{K}, n_ind, "__shared__ ufloat_t", "s_F_pb[(Nbx+2)*(Nbx+2)]");
	else
		add_variable(fileID{K}, n_ind, "__shared__ ufloat_t", "s_F_p[(Nbx+2)*(Nbx+2)*(Nbx+2)]");
		add_variable(fileID{K}, n_ind, "__shared__ ufloat_t", "s_F_pb[(Nbx+2)*(Nbx+2)*(Nbx+2)]");
	end
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
	
	
	% Place all appropriate neighbor Ids in register memory before looping over DDFs.
	COMMENT(fileID{K}, n_ind, "Load required neighbors.");
	for P = 2:l_dqs(K)
		if (sum(nbr_quad_indexed{K}(1:8) == P-1) > 0)
			add_statement(fileID{K}, n_ind, sprintf("int nbr_id_%i = cblock_ID_nbr[i_kap_b + %i*n_maxcblocks]", P-1, P-1), true);
		end
	end
	add_line(fileID{K});
	
	
	% Loop over all DDFs, ignoring index-0 DDF.
	% - Identify current p (not necessarily equal to P). Get neigbor Ids needed for DDF transfer.
	% - Reset shared memory halos to '-1'. 
	%      + Some cells will not be involved in the later write step, these are detected with '-1's that have not been converted into BC DDFs (which must be non-negative).
	% - Load DDFs in current block into shared memory.
	%      + Sync after read so all threads can access neighbors.
	%      + Load f* of p/pb into shared memory.
	% - Loop over neighbor Ids and load the DDFs that will stream into the current block.
	%      + First load for the DDFs of pb.
	%      + First load for the DDFs of p.
	% - Perform the writes after all neighbors processed. 
	%      + Unlike naive scheme, in-place streaming cannot be a pure 'pull' scheme since an exchange is always involed with neighboring blocks.
	%      + DDFs of direction p must stream into neighbors defined by nbr_ids_pb and vice-versa.
	% - Proceed with transfers across neighbor blocks.
	%      + Load conditions again.
	for P = 1:l_dqs_indexed(K)
		p = p_indexed{K}(P) + 1;
		pb_p = pb_all{K}(p) + 1;
		nbr_ids_p  = get_combos(c_all{K}(:,pb_p), c_all{K});   % Applies for DDFs p.
		nbr_ids_pb = get_combos(c_all{K}(:,p), c_all{K});   % Applies for DDFs pb.
		
		
		% Reset shared memory halos to '-1'.
		COMMENT(fileID{K}, n_ind, "");
		COMMENT(fileID{K}, n_ind, "");
		COMMENT(fileID{K}, n_ind, "");
		COMMENT(fileID{K}, n_ind, sprintf("p = %i, %i", p-1, pb_p-1));
		COMMENT(fileID{K}, n_ind, "");
		COMMENT(fileID{K}, n_ind, "");
		COMMENT(fileID{K}, n_ind, "");
		if (K == 1)
			n_ind = add_for(fileID{K}, n_ind, "int q = 0; q < 3; q++", false);
				n_ind = add_condition(fileID{K}, n_ind, "threadIdx.x + q*16 < 36", false);
					add_statement(fileID{K}, n_ind, "s_F_p[threadIdx.x + q*16] = N_Pf(-1.0)", true);
					add_statement(fileID{K}, n_ind, "s_F_pb[threadIdx.x + q*16] = N_Pf(-1.0)", true);
				n_ind = add_condition(fileID{K}, n_ind, "", true);
			n_ind = add_for(fileID{K}, n_ind, "", true);
		else
			n_ind = add_for(fileID{K}, n_ind, "int q = 0; q < 4; q++", false);
				n_ind = add_condition(fileID{K}, n_ind, "threadIdx.x + q*64 < 216", false);
					add_statement(fileID{K}, n_ind, "s_F_p[threadIdx.x + q*64] = N_Pf(-1.0)", true);
					add_statement(fileID{K}, n_ind, "s_F_pb[threadIdx.x + q*64] = N_Pf(-1.0)", true);
				n_ind = add_condition(fileID{K}, n_ind, "", true);
			n_ind = add_for(fileID{K}, n_ind, "", true);
		end
		
		
		% Load DDFs in current block into shared memory.
		add_statement(fileID{K}, n_ind, "__syncthreads()", true);
		if (K == 1)
			add_statement(fileID{K}, n_ind, sprintf("s_F_p[(I_kap+1)+(Nbx+2)*(J_kap+1)] = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + %i*n_maxcells]", p-1), true);
			add_statement(fileID{K}, n_ind, sprintf("s_F_pb[(I_kap+1)+(Nbx+2)*(J_kap+1)] = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + %i*n_maxcells]", pb_p-1), true);
		else
			add_statement(fileID{K}, n_ind, sprintf("s_F_p[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + %i*n_maxcells]", p-1), true);
			add_statement(fileID{K}, n_ind, sprintf("s_F_pb[(I_kap+1)+(Nbx+2)*(J_kap+1)+(Nbx+2)*(Nbx+2)*(K_kap+1)] = cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + %i*n_maxcells]", pb_p-1), true);
		end
		
		
		% Loop over neighbor Ids and load the DDFs that will stream into the current block.
		for id = 1:length(nbr_ids_p)
			open_a_bracket = false;
			if (sum(nbr_quad_indexed{K}(1:8) == nbr_ids_p(id)) > 0)
				COMMENT(fileID{K}, n_ind+1, sprintf("nbr %i (p = %i)", nbr_ids_p(id), p-1));
				COMMENT(fileID{K}, n_ind+1, "This nbr participates in a regular streaming exchange.");
				n_ind = add_bracket(fileID{K}, n_ind, 1, true);
				open_a_bracket = true;
				add_statement(fileID{K}, n_ind, sprintf("nbr_kap_b = nbr_id_%i", nbr_ids_p(id)), true);
			else
				%COMMENT(fileID{K}, n_ind+1, "This nbr does not participate in regular streaming exchange.");
				%n_ind = add_bracket(fileID{K}, n_ind, 1, true);
			end
			
			
			% Boundary conditions applied first. Build conditionals and the boundary condition from input.
			v_id_p = c_all{K}(:, pb_all{K}(nbr_ids_p(id)+1)+1);
			[str_cond_total_BC, str_cond_total_nBC, str_index_total_BC] = get_conditions_nbrs(v_id_p, dims(K));
			
			
			% Regular streaming (i.e. non-boundary neighbor).
			if (sum(nbr_quad_indexed{K}(1:8) == nbr_ids_p(id)) > 0)
				n_ind = add_condition(fileID{K}, n_ind, "nbr_kap_b >= 0", false);
					n_ind = add_condition(fileID{K}, n_ind, str_cond_total_nBC, false);
						add_statement(fileID{K}, n_ind, sprintf("s_F_p[%s] = cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + %i*n_maxcells]", str_index_total_BC, p-1), true);
					n_ind = add_condition(fileID{K}, n_ind, "", true);
				n_ind = add_condition(fileID{K}, n_ind, "", true);
			end
			
			
			if (open_a_bracket)
				n_ind = add_bracket(fileID{K}, n_ind, 1, false);
			end
		end
		%
		for id = 1:length(nbr_ids_pb)
			open_a_bracket = false;
			if (sum(nbr_quad_indexed{K}(1:8) == nbr_ids_pb(id)) > 0)
				COMMENT(fileID{K}, n_ind+1, sprintf("nbr %i (p = %i)", nbr_ids_pb(id), pb_p-1));
				COMMENT(fileID{K}, n_ind+1, "This nbr participates in a regular streaming exchange.");
				n_ind = add_bracket(fileID{K}, n_ind, 1, true);
				open_a_bracket = true;
				add_statement(fileID{K}, n_ind, sprintf("nbr_kap_b = nbr_id_%i", nbr_ids_pb(id)), true);
			else
				%COMMENT(fileID{K}, n_ind+1, "This nbr does not participate in regular streaming exchange.");
				%n_ind = add_bracket(fileID{K}, n_ind, 1, true);
			end
			
			
			% Boundary conditions applied first. Build conditionals and the boundary condition from input.
			v_id_pb = c_all{K}(:, pb_all{K}(nbr_ids_pb(id)+1)+1);
			[str_cond_total_BC, str_cond_total_nBC, str_index_total_BC] = get_conditions_nbrs(v_id_pb, dims(K));
			
			
			% Regular streaming (i.e. non-boundary neighbor).
			if (sum(nbr_quad_indexed{K}(1:8) == nbr_ids_pb(id)) > 0)
				n_ind = add_condition(fileID{K}, n_ind, "nbr_kap_b >= 0", false);
					n_ind = add_condition(fileID{K}, n_ind, str_cond_total_nBC, false);
						add_statement(fileID{K}, n_ind, sprintf("s_F_pb[%s] = cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + %i*n_maxcells]", str_index_total_BC, pb_p-1), true);
					n_ind = add_condition(fileID{K}, n_ind, "", true);
				n_ind = add_condition(fileID{K}, n_ind, "", true);
			end
			
			
			if (open_a_bracket)
				n_ind = add_bracket(fileID{K}, n_ind, 1, false);
			end
		end
		
		
		% Perform the writes after all neighbors processed.
		add_statement(fileID{K}, n_ind, "__syncthreads()", true);
		COMMENT(fileID{K}, n_ind+1, "Main writes within current block.");
		str_loc_p = sprintf("(I_kap + (%i) + 1) + (Nbx+2)*(J_kap + (%i) + 1)", c_all{K}(1,p), c_all{K}(2,p));
		str_loc_pb = sprintf("(I_kap + (%i) + 1) + (Nbx+2)*(J_kap + (%i) + 1)", c_all{K}(1,pb_p), c_all{K}(2,pb_p));
		if (K > 1)
			str_loc_p = [str_loc_p, sprintf(" + (Nbx+2)*(Nbx+2)*(K_kap + (%i) + 1)", c_all{K}(3,p))];
			str_loc_pb = [str_loc_pb, sprintf(" + (Nbx+2)*(Nbx+2)*(K_kap + (%i) + 1)", c_all{K}(3,pb_p))];
		end
		n_ind = add_condition(fileID{K}, n_ind, sprintf("s_F_pb[%s] >= 0", str_loc_p), false);
			add_statement(fileID{K}, n_ind, sprintf("cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + %i*n_maxcells] = s_F_pb[%s]", p-1, str_loc_p), true);
		n_ind = add_condition(fileID{K}, n_ind, "", true);
		n_ind = add_condition(fileID{K}, n_ind, sprintf("s_F_p[%s] >= 0", str_loc_pb), false);
			add_statement(fileID{K}, n_ind, sprintf("cells_f_F[i_kap_b*M_CBLOCK + threadIdx.x + %i*n_maxcells] = s_F_p[%s]", pb_p-1, str_loc_pb), true);
		n_ind = add_condition(fileID{K}, n_ind, "", true);
		
		
		% Proceed with transfers across neighbor blocks.
		for id = 1:length(nbr_ids_p)
			if (sum(nbr_quad_indexed{K}(1:8) == nbr_ids_p(id)) > 0)
				% Load conditions again.
				COMMENT(fileID{K}, n_ind+1, sprintf("Writing p = %i to nbr %i in slot pb = %i", pb_p-1, nbr_ids_p(id), p-1));
				%
				v_id_p = c_all{K}(:, pb_all{K}(nbr_ids_p(id)+1)+1);
				[str_cond_total_BC, str_cond_total_nBC, str_index_total_BC] = get_conditions_nbrs(v_id_p, dims(K));
				add_statement(fileID{K}, n_ind, sprintf("nbr_kap_b = nbr_id_%i", nbr_ids_p(id)), true);
				%
				str_loc_pb = sprintf("(I_kap + (%i) + (%i)*Nbx + 1) + (Nbx+2)*(J_kap + (%i) + (%i)*Nbx + 1)", c_all{K}(1,pb_all{K}(pb_p)+1), c_all{K}(1,nbr_ids_p(id)+1), c_all{K}(2,pb_all{K}(pb_p)+1), c_all{K}(2,nbr_ids_p(id)+1));
				if (K > 1)
					str_loc_pb = [str_loc_pb, sprintf(" + (Nbx+2)*(Nbx+2)*(K_kap + (%i) + (%i)*Nbx + 1)", c_all{K}(3,pb_all{K}(pb_p)+1), c_all{K}(3,nbr_ids_p(id)+1))];
				end
				%
				n_ind = add_condition(fileID{K}, n_ind, sprintf("(nbr_kap_b>=0) && %s", str_cond_total_nBC), false);
					n_ind = add_condition(fileID{K}, n_ind, sprintf("s_F_pb[%s] >= 0", str_loc_pb), false);
						add_statement(fileID{K}, n_ind, sprintf("cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + %i*n_maxcells] = s_F_pb[%s]", p-1, str_loc_pb), true);
					n_ind = add_condition(fileID{K}, n_ind, "", true);
				n_ind = add_condition(fileID{K}, n_ind, "", true);
			end
		end
		%
		% Proceed with transfers across neighbor blocks.
		for id = 1:length(nbr_ids_pb)
			if (sum(nbr_quad_indexed{K}(1:8) == nbr_ids_pb(id)) > 0)
				COMMENT(fileID{K}, n_ind+1, sprintf("Writing p = %i to nbr %i in slot p = %i", p-1, nbr_ids_pb(id), pb_p-1));
				%
				v_id_pb = c_all{K}(:, pb_all{K}(nbr_ids_pb(id)+1)+1);
				[str_cond_total_BC, str_cond_total_nBC, str_index_total_BC] = get_conditions_nbrs(v_id_pb, dims(K));
				add_statement(fileID{K}, n_ind, sprintf("nbr_kap_b = nbr_id_%i", nbr_ids_pb(id)), true);
				%
				str_loc_p = sprintf("(I_kap + (%i) + (%i)*Nbx + 1) + (Nbx+2)*(J_kap + (%i) + (%i)*Nbx + 1)", c_all{K}(1,pb_all{K}(p)+1), c_all{K}(1,nbr_ids_pb(id)+1), c_all{K}(2,pb_all{K}(p)+1), c_all{K}(2,nbr_ids_pb(id)+1));
				if (K > 1)
					str_loc_p = [str_loc_p, sprintf(" + (Nbx+2)*(Nbx+2)*(K_kap + (%i) + (%i)*Nbx + 1)", c_all{K}(3,pb_all{K}(p)+1), c_all{K}(3,nbr_ids_pb(id)+1))];
				end
				%
				n_ind = add_condition(fileID{K}, n_ind, sprintf("(nbr_kap_b>=0) && %s", str_cond_total_nBC), false);
					n_ind = add_condition(fileID{K}, n_ind, sprintf("s_F_p[%s] >= 0", str_loc_p), false);
						add_statement(fileID{K}, n_ind, sprintf("cells_f_F[nbr_kap_b*M_CBLOCK + threadIdx.x + %i*n_maxcells] = s_F_p[%s]", pb_p-1, str_loc_p), true);
					n_ind = add_condition(fileID{K}, n_ind, "", true);
				n_ind = add_condition(fileID{K}, n_ind, "", true);
			end
		end
		add_statement(fileID{K}, n_ind, "__syncthreads()", true);
		add_line(fileID{K});
	end

	
	% Close bracket of 'Loop over blocks in shared memory'.
	n_ind = add_for(fileID{K}, n_ind, "", true);
	n_ind = add_condition(fileID{K}, n_ind, "", true);
	
	% Close bracket of kernel header.
	n_ind = add_bracket(fileID{K}, n_ind, 1, false);
	
	
	% Build C++ routine to call streaming kernel.
	add_line(fileID{K});
	add_line(fileID{K});
	%
	args_routine = {"int i_dev", "int L"};
	args_1 = {"mesh->n_ids[i_dev][L]", "&mesh->c_id_set[i_dev][L*n_maxcblocks]", "mesh->n_maxcells", "mesh->dxf_vec[L]", "tau_vec[L]"};
	args_2 = {"mesh->c_cblock_ID_onb[i_dev]", "mesh->c_cblock_ID_nbr[i_dev]", "mesh->c_cblock_ID_nbr_child[i_dev]", "mesh->c_cblock_ID_mask[i_dev]", "mesh->n_maxcblocks"};
	args_3 = {"mesh->c_cells_f_F[i_dev]"};
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
	build_kernel_routine(fileID{K}, 0, "Solver_LBM::S", sprintf("Stream_Inpl_%s", l_dq_names{K}), args_routine, variations);
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
	printf("Streaming code: Done.\n")
else
	printf("Streaming code: Error.\n")
end
