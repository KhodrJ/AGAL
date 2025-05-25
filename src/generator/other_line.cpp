#include "gen.h"



int ProcessLineGridIndex(std::string &line)
{
	// Parameters.
	size_t n = 0;
	std::string word = "";
	std::string value = "";
	std::string line_substr = "";
	std::string line_p = "";
	std::vector<std::string> indices;
	std::vector<std::string> dims;
	std::stringstream ss;
	
	// Process grid index substitution.
	n = line.find("gGI(");
	while (n != std::string::npos)
	{
		// Get the (outer) substring enclosed by parantheses.
		line_substr = line.substr(n+4,line.size());
		line_substr = GetContentBetweenLimitChars(line_substr,'(',')');
		line_p = line_substr;
		ProcessLineMath(line_p);

		// Load the value, and the possible results to replace it with.
		ReplaceSubstring(line_p,","," ");
		ReplaceSubstring(line_p,":"," ");
		ss.clear();
		ss = std::stringstream(line_p);
		while (ss >> word)
		{
			indices.push_back(word);
			if (ss >> word)
				dims.push_back(word);
		}
		
		// Identify and perform the replacement, if valid.
		value = indices.back();
		for (int k = indices.size()-2; k >= 0; k--)
			value = indices[k] + "+" + dims[k] + "*(" + value + ")";
		ReplaceSubstring(line,"gGI("+line_substr+")",value);
		
		// Look for next gCOND.
		indices.clear();
		dims.clear();
		n = line.find("gGI(");
	}
	
	return 0;
}
