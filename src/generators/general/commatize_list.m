function [s] = commatize_list(args)
% Take a list of arguments and return a single string with commas in-between each argument.
% Input:
% - args:     List of arguments (cell array).

% Make string.
s = "";
n_args = length(args);
i = 1;
if (n_args > 1)
	for i = 1:n_args-1
		s = [s, sprintf("%s, ", args{i})];
	end
end
s = [s, sprintf("%s", args{n_args})];

endfunction
