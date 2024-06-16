function [] = build_kernel_header(fileID, n_ind, name, args, templates={})
% Build the kernel header (here, meaning its name and arguments).
% Input:
% Input:
% - fileID:       The file being updated (file ID).
% - n_ind:        Number of indentations to start with (integer).
% - name:         Number of the header (string).
% - args:         Kernel arguments (cell array of cell arrays).
% - templates:    Template arguments (cell array).

% Construct indentations.
ind = get_indent(n_ind);

% Build kernel name.
if (length(templates)>0)
	add_statement(fileID, n_ind, sprintf("template <%s>", commatize_list(templates)), false);
end
add_statement(fileID, n_ind, "__global__", false);
add_statement(fileID, n_ind, sprintf("void Cu_%s", name), false);
add_bracket(fileID, n_ind, 0, true);

% Insert arguments.
n_arg_lists = length(args);
for i = 1:n_arg_lists
	line_ender = ",";
	if (i == n_arg_lists)
		line_ender = "";
	end
	
	add_statement(fileID, n_ind+1, strcat(commatize_list(args{i}), line_ender), false);
end

% Close.
add_bracket(fileID, n_ind, 0, false);

endfunction
