function [ind] = get_indent(n_ind, ind_type=0, ind_value=1) 
% Build indentation (either tab or spaced).
% Input:
% - n_ind:       Number of indentations.
% - ind_type:    Type of indentation (0 - tab, 1 - space). I always use one tab, personally.
% - ind_value:   Number of tabs/spaces per indentation.
% Output:
% - ind:         Indentation (string).

% Parameters.
ind = "";
ind_string = "";
ind_string_base = "\t";
if (ind_type == 1)
	ind_string = " ";
end

% Multiply if 'ind_value' greater than one.
for i = 1:ind_value
	ind_string = strcat(ind_string, ind_string_base);
end

% Make indentation.
for i = 1:n_ind
	ind = strcat(ind,ind_string);
end

endfunction
