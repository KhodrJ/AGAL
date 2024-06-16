function [n_ind_new] = add_bracket(fileID, n_ind, bracket_type, bracket_open, add_semi=false)
% Insert a bracket (parantheses or curly) and updated indentation tracker accordingly.
% Input:
% - fileID:         The file being updated (file ID).
% - n_ind:          Number of indentations to start with (integer).
% - bracket_type:   Parantheses (0) or curly (1) (integer).
% - bracket_open:   Are brackets opening? (boolean).
% - add_semi:       Add a semi-colon at the end? (boolean).
% Output:
% - n_ind_new:      Updated n_ind.

b = "";

if (bracket_open)
	if (bracket_type == 0)
		b = "(";
	else
		b = "{";
	end
	
	n_ind_new = n_ind+1;
	add_statement(fileID, n_ind, sprintf("%s", b), false);
else
	if (bracket_type == 0)
		b = ")";
	else
		b = "}";
	end
	
	n_ind_new = n_ind-1;
	add_statement(fileID, n_ind_new, sprintf("%s", b), add_semi);
end

endfunction
