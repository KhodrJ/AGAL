function [] = interp_linear_tree(fileID, n_ind, n_d, vars_s1, vars_s2, conds="", statements={})
% Linear interpolation in either 2D or 3D on a sample 4x4(x4) space of unit area/volume.
% Input:
% - fileID:        The file being updated (file ID).
% - n_ind:         Number of indentations to start with (integer).
% - n_d:           Number of dimensions (integer).
% - vars_s1:       The variables being interpolated (cell array).
% - vars_s2:       The destination variables (cell array).
% - conds:         Condition to precede interpolation which may depend on child index (string).
% - statements:    Additional statements to follow within the supplied condition (string).

if (n_d == 2)
	% First loop to build indices.
	indices = zeros(4,4);
	for j = 1:4
		for i = 1:4
			indices(i,j) = (i-1)+4*(j-1);
		end
	end
	
	% Build children sequentially.
	index_set = {...
		indices(1:2,1:2), indices(3:4,1:2), indices(1:2,3:4), indices(3:4,3:4) ...
	};
	for q = 0:4-1
		COMMENT(fileID, n_ind+1, sprintf("Child %i", q));
		
		if (~strcmp(conds,""))
			n_ind = add_condition(fileID, n_ind, sprintf(conds,q), false);
		end
		
		for p = 1:length(vars_s1)
			interp_linear(fileID, n_ind, 2, index_set{q+1}, sprintf(vars_s1{p}, q), vars_s2{p});   % Child q.
		end
		for p = 1:length(statements)
			add_statement(fileID, n_ind, sprintf(statements{p}, q), true);
		end
		
		if (~strcmp(conds,""))
			n_ind = add_condition(fileID, n_ind, "", true);
		end
	end
	
	
	%print_interp_linear(fileID, n_ind, 2, indices(3:4,1:2), sprintf(var_s1, 1), var_s2);   % Child 1.
	%print_interp_linear(fileID, n_ind, 2, indices(1:2,3:4), sprintf(var_s1, 2), var_s2);   % Child 2.
	%print_interp_linear(fileID, n_ind, 2, indices(3:4,3:4), sprintf(var_s1, 3), var_s2);   % Child 3.

else
	% First loop to build indices.
	indices = zeros(4,4,4);
	for k = 1:4
		for j = 1:4
			for i = 1:4
				indices(i,j,k) = (i-1)+4*(j-1)+4*4*(k-1);
			end
		end
	end
	
	% Build children sequentially.
	index_set = {...
		indices(1:2,1:2,1:2), indices(3:4,1:2,1:2), indices(1:2,3:4,1:2), indices(3:4,3:4,1:2), ...
		indices(1:2,1:2,3:4), indices(3:4,1:2,3:4), indices(1:2,3:4,3:4), indices(3:4,3:4,3:4) ...
	};
	for q = 0:8-1
		COMMENT(fileID, n_ind+1, sprintf("Child %i", q));
		
		if (~strcmp(conds,""))
			n_ind = add_condition(fileID, n_ind, sprintf(conds,q), false);
		end
		
		for p = 1:length(vars_s1)
			interp_linear(fileID, n_ind, 3, index_set{q+1}, sprintf(vars_s1{p}, q), vars_s2{p});   % Child q.
		end
		for p = 1:length(statements)
			add_statement(fileID, n_ind, sprintf(statements{p}, q), true);
		end
		
		if (~strcmp(conds,""))
			n_ind = add_condition(fileID, n_ind, "", true);
		end
	end
	
	
	
	%print_interp_linear(fileID, n_ind, 3, indices(1:2,1:2,1:2), sprintf(var_s1, 0), var_s2);   % Child 0.
	%print_interp_linear(fileID, n_ind, 3, indices(3:4,1:2,1:2), sprintf(var_s1, 1), var_s2);   % Child 1.
	%print_interp_linear(fileID, n_ind, 3, indices(1:2,3:4,1:2), sprintf(var_s1, 2), var_s2);   % Child 2.
	%print_interp_linear(fileID, n_ind, 3, indices(3:4,3:4,1:2), sprintf(var_s1, 3), var_s2);   % Child 3.
	%print_interp_linear(fileID, n_ind, 3, indices(1:2,1:2,3:4), sprintf(var_s1, 4), var_s2);   % Child 4.
	%print_interp_linear(fileID, n_ind, 3, indices(3:4,1:2,3:4), sprintf(var_s1, 5), var_s2);   % Child 5.
	%print_interp_linear(fileID, n_ind, 3, indices(1:2,3:4,3:4), sprintf(var_s1, 6), var_s2);   % Child 6.
	%print_interp_linear(fileID, n_ind, 3, indices(3:4,3:4,3:4), sprintf(var_s1, 7), var_s2);   % Child 7.
end

endfunction
