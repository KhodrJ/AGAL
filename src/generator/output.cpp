#include "gen.h"

std::string make_indent(int indents, std::string tab="    ")
{
	std::string t = "";
	for (int k = 0; k < indents; k++)
		t += tab;
	
	return t;
}

int ProcessLineFormatInterpret(ImportedSet &iset, std::string set_label, std::map<std::string,std::string> &fparams, std::string &line, int &indent, std::vector<int> &open_fors)
{
	// Parameters.
	size_t N = 0;
	int n = 0;
	int index_size = 1;
	std::string word = "";
	std::string word2 = "";
	std::string loop_index = "";
	std::string limit_m = "";
	std::string limit_M = "";
	std::string limit_incr = "";
	std::vector<std::string> loop_indices;
	std::stringstream ss(line);
	ss >> word;
	
	// For-loop.
	if (word == "OUTFOR")
	{
		// Process the original line.
		ProcessLoopStatements(iset,set_label,line);
		ss.clear();
		ss = std::stringstream(line);
		ss >> word;
		
		line = "";
		ss >> loop_index >> index_size;
		n = loop_index.length()/index_size;
		
		// Collect the index substrings from loop_index and their corresponding limits.
		for (int j = 0; j < n; j++)
		{
			loop_indices.push_back(loop_index.substr(j*index_size,index_size));
			ss >> limit_m >> limit_M >> limit_incr;
			
			line += make_indent(indent) + "for (int " + loop_indices[j] + " = " + limit_m + "; " + loop_indices[j] + " < " + limit_M + "; " + loop_indices[j] + " += " + limit_incr + ")\n" + make_indent(indent) + "{\n";
			indent++;
		}

		// Store the number of brackets to close when encountering END_OUTFOR.
		open_fors.push_back(n);
	}
	else if (word == "END_OUTFOR")
	{
		if (open_fors.size() > 0)
		{
			line = "";
			n = open_fors.back();
			open_fors.pop_back();
			
			for (int j = 0; j < n; j++)
			{
				indent--;
				line += make_indent(indent) + "}\n";
			}
		}
		else
			line = "(err:MORE END_OUTFORS THAN OUTFORS)\n";
	}
	
	// If-statement.
	else if (word == "OUTIF" || word == "OUTIFL")
	{
		// Process the original line.
		word2 = word;
		ProcessLoopStatements(iset,set_label,line);
		ss.clear();
		ss = std::stringstream(line);
		ss >> word;
		line = "";
		
		while (ss >> word)
			line += word + " ";
		line = line.substr(0,line.length()-1);
		
		if (word2 == "OUTIF")
		{
			line = make_indent(indent) + "if " + line + "\n" + make_indent(indent) + "{\n";
			indent++;
		}
		else
		{
			line = make_indent(indent) + "if " + line + "\n";
			indent++;
		}
	}
	else if (word == "OUTELSE")
	{
		indent--;
		line = make_indent(indent) + "}\n" + make_indent(indent) + "else\n" + make_indent(indent) + "{\n";
		indent++;
	}
	else if (word == "END_OUTIF")
	{
		indent--;
		line = make_indent(indent) + "}\n";
	}
	else if (word == "END_OUTIFL")
	{
		indent--;
		line = "";
	}
	
	// Other special lines.
	else if (word == "<")
	{
		line =  make_indent(indent) + "\n";
	}
	
	// Output file parameters.
	else if (word == "FILE_NAME")
	{
		ss >> word;
		fparams["name"] = word;
		line = "";
	}
	else if (word == "FILE_DIR")
	{
		ss >> word;
		fparams["dir"] = word;
		line = "";
	}
	
	// Regular statement.
	else
	{
		if (line != "\n")
			line = make_indent(indent) + line;
	}
	
	return 0;
}

int Process_OutputFormatting(ImportedSet &iset, std::string set_label, std::map<std::string,std::string> &fparams, std::string &s_in)
{
	// Parameters.
	int indent = 1;
	std::string line = "";
	std::vector<int> open_fors;
	std::stringstream ss = std::stringstream(s_in);
	
	s_in = "";
	while (ss.peek() != EOF)
	{
		//line = CollectLineFormatted(ss);
		line = CollectLine(ss);
		ProcessLineFormatInterpret(iset,set_label,fparams,line,indent,open_fors);
		s_in += line;
	}
	
	return 0;
}
