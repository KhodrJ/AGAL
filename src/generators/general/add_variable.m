function [] = add_variable(fileID, n_ind, keywords, name)
% Declare a new variable (this is mostly for clean-up).
% Input:
% - fileID:     The file being updated (file ID).
% - n_ind:      Number of indentations to start with (integer).
% - keywords:   Keywords to appear before variable name such as type (string).
% - name:       Name of variable (string).

% Construct indentations.
ind = get_indent(n_ind);

% Print.
fprintf(fileID, "%s%s %s;\n", ind, keywords, name);

endfunction
