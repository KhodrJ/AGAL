function [] = add_string(fileID, s)
% Insert a new line (this is mostly for clean-up).
% Input:
% - fileID:     The file being updated (file ID).
% - s:          Input string (string).

fprintf(fileID, s);

endfunction
