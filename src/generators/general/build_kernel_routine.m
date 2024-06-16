function [] = build_kernel_routine(fileID, n_ind, obj_type, name, args_routine, variations={})
% Build the kernel header (here, meaning its name and arguments).
% Input:
% Input:
% - fileID:         The file being updated (file ID).
% - n_ind:          Number of indentations to start with (integer).
% - name:           Indicator for the object that this routine is associated with (string).
% - name:           Number of the header (string).
% - args_routine:   Arguments to C++ routine (cell array).
% - args_kernel:    Arguments to CUDA kernel (cell array of cell arrays).
% - variations:     List of variations in conditions, templates etc. (cell array of cell arrays).
%
% Variation labels (must be in order):
% - condition:      The condition to apply before launching kernel.
% - params:         Launch parameters.
% - arguments:      Kernel arguments.
% - template:       Template arguments.

% Construct indentations.
ind = get_indent(n_ind);

% Build kernel name.
add_statement(fileID, n_ind, sprintf("int %s_%s(%s)", obj_type, name, commatize_list(args_routine)), false);
n_ind = add_bracket(fileID, n_ind, 1, true);

% Add kernels and their variations.
n_varies = length(variations);
for j = 1:n_varies
	% Get current variation parameters.
	variation_j = variations{j};
	kernel_condition_j = variation_j{2};
	kernel_params_j = variation_j{4};
	kernel_args_j = variation_j{6};
	kernel_template_j = variation_j{8};
	
	% Print condition.
	n_ind = add_condition(fileID, n_ind, kernel_condition_j, false);
	
	% Print kernel name and launch parameters.
	add_statement(fileID, n_ind, sprintf("Cu_%s%s<<<%s>>>", name, kernel_template_j, kernel_params_j), false);
	n_ind = add_bracket(fileID, n_ind, 0, true);
		n_arg_lists = length(kernel_args_j);
		for i = 1:n_arg_lists
			line_ender = ",";
			if (i == n_arg_lists)
				line_ender = "";
			end
			
			add_statement(fileID, n_ind, strcat(commatize_list(kernel_args_j{i}), line_ender), false);
		end
	n_ind = add_bracket(fileID, n_ind, 0, false, true);
	
	% Close condition.
	n_ind = add_condition(fileID, n_ind, "", true);
end

% Add return statement.
add_line(fileID);
add_statement(fileID, n_ind, "return 0", true);

% Close.
n_ind = add_bracket(fileID, n_ind, 1, false);

endfunction
