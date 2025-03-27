#include "gen.h"

// Process the in-for and in-if loops recursively.
int ProcessLineIfAndFor(ImportedSet &iset, std::string set_label, std::istream &in, std::string &line)
{
	std::string first_word = "";
	std::stringstream ss(line);
	ss >> first_word;
	
	// If first word is INFOR, then unroll a for-loop in the generator.
	if (first_word == "INFOR")
		ProcessFor(iset, set_label, in, line);
	
	// If first word is INIF, then unroll an in-if statement.
	if (first_word == "INIF")
		ProcessIf(iset, set_label, in, line);
	
	return 0;
}

int ProcessLoopStatements(ImportedSet &iset, std::string set_label, std::string &line)
{
	std::string line_p = "";
	while (line_p != line)
	{
		line_p = line;
		ApplyAllOperations(iset, set_label, line);
	}
	
	return 0;
}

int ProcessFor(ImportedSet &iset, std::string set_label, std::istream &in, std::string &line)
{
	// Parameters.
	int for_counter = 1;
	int endfor_counter = 0;
	std::string word = "";
	std::string line_p = "";
	std::string looped_section = "";
	std::string interm_looped_section = "";
	std::string final_looped_section = "";
	
	// Repeatedly apply mathematical and variable substitutions until convergence.
	ProcessLoopStatements(iset, set_label, line);
	
	// Get the loop indices and limits. Place them in vectors.
	//ApplyAllOperations(iset, set_label, in, line);
	std::string collected_indices = "";          // e.g. ijk, one word
	std::vector<std::string> loop_indices;       // e.g. 0: i, 1: j, 2: k
	int index_size = 1;                          // e.g. length of 'i'
	std::vector<int> loop_limits;                // e.g. 0 5 1 => (int i = 0; i < 5; i += 1)
	std::stringstream ss(line);
	ss >> collected_indices >> collected_indices >> index_size; // Note: discarding the first word (INFOR).
	int K = collected_indices.size() / index_size;
	for (int k = 0; k < K; k++)
	{
		loop_indices.push_back( collected_indices.substr(k*index_size, index_size) );
		int limit_m, limit_M, limit_p;
		ss >> limit_m >> limit_M >> limit_p;
		loop_limits.push_back(limit_m);
		loop_limits.push_back(limit_M);
		loop_limits.push_back(limit_p);
	}
	
	// Collect the section of code to be looped over.
	// To capture inner loops, count how many times we go over "INFOR" and "END_INFOR". They should match at the end of the outer loop.
	while (for_counter != 2*endfor_counter)
	{
		line_p = CollectLine(in);
		
		if (line_p.find("INFOR") != std::string::npos)
			for_counter++;
		if (line_p.find("END_INFOR") != std::string::npos)
			endfor_counter++;
		
		if (for_counter != 2*endfor_counter)
			looped_section += line_p;
	}
	
	
	// Perform the substitutions of indices in the looped section.
	for (int k = 0; k < K; k++)
	{
		// Duplicate the looped section according to each set of loop limits.
		int limit_m = loop_limits[0+3*k];
		int limit_M = loop_limits[1+3*k];
		int limit_p = loop_limits[2+3*k];
		if (k > 0)
			looped_section = final_looped_section;
		final_looped_section = "";
		for (int j = limit_m; j < limit_M; j++)
		{
			interm_looped_section = looped_section;
			ReplaceSubstring(interm_looped_section, "<"+loop_indices[k]+">", std::to_string(j));
			final_looped_section += interm_looped_section;
		}
	}
	
	// Go back through the final output (after substitutions are done) and process the lines.
	ss.clear();
	ss = std::stringstream(final_looped_section);
	final_looped_section = "";
	while (ss.peek() != EOF)
	{
		line_p = CollectLine(ss);
		ProcessLineIfAndFor(iset, set_label, ss, line_p);
		final_looped_section += line_p;
	}
	
	// Now update the line.
	line = final_looped_section;
	
	return 0;
}

int ProcessIf(ImportedSet &iset, std::string set_label, std::istream &in, std::string &line)
{
	// Parameters.
	int t = 0;
	bool passed_else = false;
	int if_counter = 1;
	int else_counter = 0;
	int endif_counter = 0;
	std::string word = "";
	std::string line_p = "";
	std::string included_section = "";
	std::string other_included_section = "";
	std::string final_included_section = "";
	
	// Repeatedly apply mathematical and variable substitutions until convergence.
	ProcessLoopStatements(iset, set_label, line);
	
	// Extract the conditional statement.
	//ApplyAllOperations(iset, set_label, in, line);
	std::stringstream ss(line);
	ReplaceSubstring(line, "INIF", "");
	
	// Collect the section of code to be included if the condition is true.
	while (if_counter != 2*endif_counter)
	{
		line_p = CollectLine(in);
		
		if (line_p.find("INIF") != std::string::npos)
			if_counter++;
		if (line_p.find("INELSE") != std::string::npos)
			else_counter++;
		if (line_p.find("END_INIF") != std::string::npos)
			endif_counter++;
		
		if (!passed_else && (if_counter-endif_counter == else_counter))
		{
			passed_else = true;
			continue;
		}
		
		if (if_counter != 2*endif_counter)
		{
			if (!passed_else)
				included_section += line_p;
			else
				other_included_section += line_p;
		}
	}
	
	// Process the condition, then choose which block of code to include.
	if (parser.compile(line,expression))
	{
		t = expression.value();
		if (t)
			final_included_section = included_section;
		if (!t)
			final_included_section = other_included_section;
	
		// Go back through the final output (after substitutions are done) and process the lines.
		ss.clear();
		ss.clear();
		ss = std::stringstream(final_included_section);
		final_included_section = "";
		while (ss.peek() != EOF)
		{
			line_p = CollectLine(ss);
			ProcessLineIfAndFor(iset, set_label, ss, line_p);
			final_included_section += line_p;
		}
		
		// Now update the line.
		line = final_included_section;
	}
	else
	{
		std::cout << line << std::endl;
		line = "ERROR, DID NOT COMPILE: " + line;
	}
	
	return 0;
}

int Process_OutputInLoops(ImportedSet &iset, std::string set_label, std::string &s_in)
{
	// Parameters.
	std::string line = "";
	std::stringstream ss = std::stringstream(s_in);
	
	s_in = "";
	while (ss.peek() != EOF)
	{
		line = CollectLine(ss);
		ProcessLineIfAndFor(iset, set_label, ss, line);
		s_in += line;
	}
	
	return 0;
}
