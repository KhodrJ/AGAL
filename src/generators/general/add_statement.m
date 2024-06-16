function [] = add_statement(fileID, n_ind, name, add_semi)
% Declare a new statement (this is mostly for clean-up).
% Input:
% - fileID:     The file being updated (file ID).
% - n_ind:      Number of indentations to start with (integer).
% - name:       The statement to be added (string).
% - add_semi:   Add a semi-colon at the end? (boolean).

% Construct indentations.
ind = get_indent(n_ind);

% Print.
line_ender = "";
if (add_semi)
	line_ender = ";";
end
fprintf(fileID, "%s%s%s\n", ind, name, line_ender);

endfunction
