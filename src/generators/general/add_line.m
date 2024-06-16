function [] = add_line(fileID)
% Insert a new line (this is mostly for clean-up).
% Input:
% - fileID:     The file being updated (file ID).

fprintf(fileID, "\n");

endfunction
