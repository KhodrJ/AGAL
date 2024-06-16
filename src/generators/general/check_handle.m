function [b] = check_handle(f, args)
% Verify existence/validity of function by testing sample arguments.
% Input:
% - f:      Function handle (handle).
% - args:   Arguments to function for testing (array).

%fh = functions(f);
%fh.file
%b = ~isempty(fh.file);

cval = 1;
try
	f(args);
catch
	printf("Warning: invalid arguments pass to function %s...\n", func2str(f))
	cval = 0;
end_try_catch

b = cval;

endfunction
