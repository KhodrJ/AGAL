function [] = COMMENT(fileID, n_ind, comment)
% Print a comment (this is mainly for visibility and clean-up).
% Input:
% - fileID:     The file being updated (file ID).
% - n_ind:      Number of indentations to start with (integer).
% - comment:    The comment to be printed.

add_statement(fileID, n_ind, sprintf("// %s", comment), false);
 
endfunction
