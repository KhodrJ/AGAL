function [n_ind_new] = add_for(fileID, n_ind, statement, close_for)
% Insert (or close) a for-loop and update the indentation tracker accordingly.
% Input:
% - fileID:           The file being updated (file ID).
% - n_ind:            Number of indentations to start with (integer).
% - statement:        The statement to be added (string).
% - close_for         Close loop? (boolean).

if (~close_for)
	% Construct indentations.
	ind = get_indent(n_ind);
	
	% Print.
	fprintf(fileID, "%sfor (%s)\n", ind, statement);
	fprintf(fileID, "%s{\n", ind);

	% Update tracker.
	n_ind_new = n_ind+1;
else
	% Update tracker.
	n_ind_new = n_ind-1;

	% Construct indentations.
	ind = get_indent(n_ind_new);

	fprintf(fileID, "%s}\n", ind);
end

endfunction
