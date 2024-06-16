function [n_ind_new] = add_condition(fileID, n_ind, condition, close_condition)
% Insert (or close) a condition and update the indentation tracker accordingly.
% Input:
% - fileID:           The file being updated (file ID).
% - n_ind:            Number of indentations to start with (integer).
% - condition:        The condition to be added (string).
% - close_condition   Close condition? (boolean).

if (~close_condition)
	% Construct indentations.
	ind = get_indent(n_ind);
	
	% Print.
	fprintf(fileID, "%sif (%s)\n", ind, condition);
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
