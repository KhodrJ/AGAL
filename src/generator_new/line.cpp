#include "gen.h"

// Apply operations to the line such as:
// - Processing logical and mathematical expressions with the exprtk package.
// - Replacing vector/matrix elements with imported values.
// This is done by applying flags after the REG keyword. The following flags are used:
// - m:       Resolve logical/mathetmatic expressions.
// - v:       Substitute a vector/matrix element in the line
int ProcessLineRegular(ImportedSet &iset, std::string set_label, std::istream &in, std::string &line)
{
	// Parameters.
	size_t i_flags = 0;
	int n_flags = 0;
	std::string flags = "";
	std::stringstream ss(line);
	std::string first_word = "";
	ss >> first_word;
	
	if (first_word.substr(0,3) == "REG")
	{
		// Processing is only needed if there actually are flags.
		i_flags = first_word.find("-");
		if (i_flags != std::string::npos)
		{
			flags = first_word.substr(i_flags+1,first_word.size());
			n_flags = flags.size();
			for (int k = 0; k < n_flags; k++)
			{
				if (flags[k] == 's')
					ProcessLineSumsAndProds(line);
				if (flags[k] == 'v')
					ProcessLineSetLabels(iset, set_label, line);
				if (flags[k] == 'm')
					ProcessLineMath(line);
				if (flags[k] == 'z')
					ProcessLineGetRidOfZerosSum(line);
				if (flags[k] == 'c')
					ProcessLineConditionConstructor(line);
				if (flags[k] == 'a')
					ProcessLineSimplifyAndsOrs(line);
				if (flags[k] == 'o')
					ProcessLineGridIndex(line);
			}
		}
		ReplaceSubstring(line,first_word,"");
	}
	
	return 0;
}

int ApplyAllOperations(ImportedSet &iset, std::string set_label, std::string &line)
{
	ProcessLineSetLabels(iset, set_label, line);
	ProcessLineMath(line);
	ProcessLineSumsAndProds(line);
	ProcessLineGetRidOfZerosSum(line);
	ProcessLineConditionConstructor(line);
	ProcessLineSimplifyAndsOrs(line);
	
	return 0;
}

int ProcessLineSetLabels(ImportedSet &iset, std::string set_label, std::string &line, int var)
{
	// Parameters.
	size_t n;
	std::string name = "";
	std::string name_subtr = "";
	std::string name_subtr_o = "";
	std::string namewset = "";
	std::string type = "";
	
	// Go through the possible values (by using the list of imported data names).
	for (int k = 0; k < iset.names.size(); k++)
	{
		name = iset.names[k];
		n = line.find(name);
		type = iset.imported_types[name];
		if (var == 1 || (var != 1 && (type=="value" || type=="setvalue")))
		{
			while (n != std::string::npos)
			{
				namewset = name + set_label;
				
				if (type == "value")
				{
					ReplaceSubstring(line, name, iset.imported_values[name]);
				}
				if (type == "setvalue")
				{
					ReplaceSubstring(line, name, iset.imported_setvalues[namewset]);
				}
				if (var == 1) // If var != 1, just replace value/setvalues, which do not require specification of indices.
				{
					if (type == "vector")
					{
						name_subtr = line.substr(n+name.size()+1,line.size());
						name_subtr = "("+GetContentBetweenLimitChars(name_subtr,'(',')')+")";
						name_subtr_o = name_subtr;
						ProcessLineSetLabels(iset,set_label,name_subtr);
						ProcessLineMath(name_subtr);
						int t = 1;
						if (parser.compile(name_subtr,expression))
						{
							t = expression.value();
							if (t < iset.imported_vectors[name].size() && t >= 0)
								ReplaceSubstring(line, name+name_subtr_o, iset.imported_vectors[name][t]);
							else
							{
								n = line.find(name,n+1);
								continue;
							}
								//ReplaceSubstring(line, name+name_subtr_o, "(err:INVALID INDEX)");
						}
						else
						{
							n = line.find(name,n+1);
							continue;
						}
							//ReplaceSubstring(line, name+name_subtr_o, "(err:INVALID INDEX)");
					}
					if (type == "setvector")
					{
						name_subtr = line.substr(n+name.size()+1,line.size());
						name_subtr = "("+GetContentBetweenLimitChars(name_subtr,'(',')')+")";
						name_subtr_o = name_subtr;
						ProcessLineSetLabels(iset,set_label,name_subtr);
						ProcessLineMath(name_subtr);
						int t = 1;
						if (parser.compile(name_subtr,expression))
						{
							t = expression.value();
							if (t < iset.imported_setvectors[namewset].size() && t >= 0)
								ReplaceSubstring(line, name+name_subtr_o, iset.imported_setvectors[namewset][t]);
							else
							{
								n = line.find(name,n+1);
								continue;
							}
								//ReplaceSubstring(line, name+name_subtr_o, "(err:INVALID INDEX)");
						}
						else
						{
							n = line.find(name,n+1);
							continue;
						}
							//ReplaceSubstring(line, name+name_subtr_o, "(err:INVALID INDEX)");
					}
				}
				
				n = line.find(name);
			}
		}
	}
	
	return 0;
}

int ProcessLineMath(std::string &line)
{
	// Parameters.
	size_t n = 0;
	int ti = 0;
	double td = 0.0;
	std::string line_substr = "";
	
	// Process integer computation.
	n = line.find("gI(");
	while (n != std::string::npos)
	{
		line_substr = line.substr(n+3,line.size());
		line_substr = GetContentBetweenLimitChars(line_substr,'(',')');
		ti = 1;
		if (parser.compile(line_substr,expression))
		{
			ti = expression.value();
			ReplaceSubstring(line, "gI("+line_substr+")", std::to_string(ti));
			n = line.find("gI(");
		}
		else
			n = line.find("gI(",n+1);
	}
	
	// Process double-precision computation.
	n = line.find("gD(");
	while (n != std::string::npos)
	{
		line_substr = line.substr(n+3,line.size());
		line_substr = GetContentBetweenLimitChars(line_substr,'(',')');
		td = 1;
		if (parser.compile(line_substr,expression))
		{
			td = expression.value();
			std::sprintf(c_buff,"%17.15f",td);
			ReplaceSubstring(line, "gD("+line_substr+")", std::string(c_buff));
			n = line.find("gD(");
		}
		else
			n = line.find("gD(",n+1);
	}
	
	return 0;
}







int ProcessLineSumsAndProds(std::string &line)
{
	// Parameters.
	size_t n1 = 0;
	size_t n2 = 0;
	size_t com1 = 0;
	size_t com2 = 0;
	int limit_m = 0;
	int limit_M = 0;
	int limit_incr = 0;
	std::string line_substr = "";
	std::string index = "";
	std::string limits = "";
	std::string content = "";
	std::string content_p = "";
	std::string expanded = "";
	std::stringstream ss;
	
	// Process integer computation.
	n1 = line.find("gSUM(");
	n2 = line.find("gPROD(");
	while (n1 != std::string::npos || n2 != std::string::npos)
	{
		// Get the (outer) substring enclosed by parantheses.
		if (n1 < n2)
			line_substr = line.substr(n1+5,line.size());
		if (n1 > n2)
			line_substr = line.substr(n2+6,line.size());
		line_substr = GetContentBetweenLimitChars(line_substr,'(',')');
		com1 = line_substr.find(",");
		com2 = line_substr.find(",",com1+1);
		if (com1 == com2 || (com1 == std::string::npos || com2 == std::string::npos))
		{
			if (n1 < n2)
				ReplaceSubstring(line,"gSUM("+line_substr+")","(err:INVALID SUM)");
			if (n1 > n2)
				ReplaceSubstring(line,"gPROD("+line_substr+")","(err:INVALID PROD)");
			break;
		}
		
		// Get the iterations, limits, and content to be expanded.
		index = line_substr.substr(0,com1);
		limits = line_substr.substr(com1+1,com2-(com1+1));
		content = line_substr.substr(com2+1,line_substr.size()-(com2+1));
		
		ProcessLineMath(limits);
		ss.clear();
		ss = std::stringstream(index);
		ss >> index;
		ss.clear();
		ss = std::stringstream(limits);
		ss >> limit_m >> limit_M >> limit_incr;
		
		// Expand the content.
		for (int i = limit_m; i < limit_M; i += limit_incr)
		{
			content_p = content;
			ReplaceSubstring(content_p,"<"+index+">",std::to_string(i));
			if (i > limit_m)
			{
				if (n1 < n2)
					content_p = "+" + content_p;
				if (n1 > n2)
					content_p = "*" + content_p;
			}
			expanded += content_p;
		}
		ProcessLineMath(expanded);
		if (n1 < n2)
			ReplaceSubstring(line,"gSUM("+line_substr+")",expanded);
		if (n1 > n2)
			ReplaceSubstring(line,"gPROD("+line_substr+")",expanded);
		expanded = "";
		
		// Look for next gSUM/gPROD.
		n1 = line.find("gSUM(");
		n2 = line.find("gPROD(");
	}
	
	return 0;
}



int ProcessLineGetRidOfZerosSum(std::string &line)
{
	// Parameters.
	bool found_zero = false;
	int neg_counter = 0;
	size_t n = 0;
	double t = 0.0;
	std::string line_p = "";
	std::string line_q = "";
	std::string line_t = "";
	std::string line_substr = "";
	std::vector<std::string> vec;
	std::vector<std::string> vec2;
	std::vector<std::string> fvec;
	std::vector<std::string> fvecn;
	
	// Process integer computation.
	n = line.find("gNz(");
	while (n != std::string::npos)
	{
		// Get the substring enclosed by parantheses.
		line_substr = line.substr(n+4,line.size());
		line_substr = GetContentBetweenLimitChars(line_substr,'(',')');
	
		// Remove zeros in the expression (A1*x + A2*x2 + ... + A3*x3).
		line_p = line_substr;
		ReplaceSubstring(line_p," ","");
		ReverseCommatize(line_p, vec, "+");
		for (int i = 0; i < vec.size(); i++)
		{
			// Reset.
			vec2.clear();
			found_zero = false;
			neg_counter = 0;
			
			// Break each term in the sum up.
			line_p = vec[i];
			ReverseCommatize(line_p, vec2, "*");
			for (int k = 0; k < vec2.size(); k++)
			{
				line_q = vec2[k];
				ReplaceSubstring(line_q,"(ufloat_t)","");
				ReplaceSubstring(line_q,"(","");
				ReplaceSubstring(line_q,")","");
				
				// Check if either term is zero.
				if (parser.compile(line_q,expression))
				{
					t = expression.value();
					
					if (t == 0.0)
					{
						found_zero = true;
					}
					if (t == 1.0)
					{
						if (vec2.size()>1)
						{
							vec2.erase(vec2.begin()+k);
							k--;
						}
					}
					if (t < 0)
					{
						neg_counter++;
						if (t == -1.0 && vec2.size() > 1)
						{
							vec2.erase(vec2.begin()+k);
							k--;
						}
					}
				}
			}
			if (found_zero)
				continue;
			else
			{
				if (neg_counter%2 == 0)
					fvec.push_back(Commatize(vec2,"*"));
				else
					fvecn.push_back(Commatize(vec2,"*"));
			}
		}
		
		// Update line.
		line_p = "";
		if (fvec.size() > 0)
			line_p += "("+Commatize(fvec,"+")+")";
		if (fvecn.size() > 0)
			line_p += "-("+Commatize(fvecn,"+")+")";
		if (fvec.size() == 0 && fvecn.size() == 0)
			line_p = "(ufloat_t)(0.0)";
		ReplaceSubstring(line, "gNz("+line_substr+")", line_p);
		vec.clear();
		fvec.clear();
		fvecn.clear();
		n = line.find("gNz(");
	}
	
	return 0;
}






int ProcessLineConditionConstructor( std::string &line)
{
	// Parameters.
	size_t n = 0;
	std::string word = "";
	std::string line_substr = "";
	std::string line_p = "";
	std::string value;
	std::vector<std::string> ifs;
	std::vector<std::string> thens;
	std::stringstream ss;
	
	// Process integer computation.
	n = line.find("gCOND(");
	while (n != std::string::npos)
	{
		// Get the (outer) substring enclosed by parantheses.
		line_substr = line.substr(n+6,line.size());
		line_substr = GetContentBetweenLimitChars(line_substr,'(',')');
		line_p = line_substr;
		ProcessLineMath(line_p);
		ProcessLineConditionConstructor(line_p);

		// Load the value, and the possible results to replace it with.
// 		ReplaceSubstring(line_p,","," ");
// 		ReplaceSubstring(line_p,":"," ");
		ReplaceStringsBetweenLimitStrings(line_p,","," ","(",")");
		ReplaceStringsBetweenLimitStrings(line_p,":"," ","(",")");
		ss.clear();
		ss = std::stringstream(line_p);
		ss >> value;
		while (ss >> word)
		{
			ifs.push_back(word);
			ss >> word;
			thens.push_back(word);
		}
		
		// Identify and perform the replacement, if valid.
		word = "";
		for (int k = 0; k < ifs.size(); k++)
		{
			if (value == ifs[k])
				word = thens[k];
			
			if (ifs[k] == "def." && word == "")
				word = thens[k];
		}
		//if (word == "")
		//	ReplaceSubstring(line,"gCOND("+line_substr+")","(err:INVALID COND)");
		//else
		ReplaceSubstring(line,"gCOND("+line_substr+")",word);
		
		// Look for next gCOND.
		ifs.clear();
		thens.clear();
		n = line.find("gCOND(");
	}
	
	return 0;
}




int ProcessLineSimplifyAndsOrs( std::string &line)
{
	// Parameters.
	bool found_false = false;
	bool found_true = false;
	size_t n = 0;
	std::string word = "";
	std::string line_substr = "";
	std::string line_p = "";
	std::string value;
	std::vector<std::string> terms;
	std::stringstream ss;
	
	// Simplify 'ands'.
	n = line.find("gNa(");
	while (n != std::string::npos)
	{
		// Get the (outer) substring enclosed by parantheses.
		line_substr = line.substr(n+4,line.size());
		line_substr = GetContentBetweenLimitChars(line_substr,'(',')');
		line_p = line_substr;

		// Load the terms in the conditional.
		ReplaceStringsBetweenLimitStrings(line_p," ","","(",")");
		ReplaceStringsBetweenLimitStrings(line_p,"and"," ","(",")");
		ReplaceStringsBetweenLimitStrings(line_p,"&&"," ","(",")");
		ReplaceSubstring(line_p,"(true)","true");
		ReplaceSubstring(line_p,"(false)","false");
		ss.clear();
		ss = std::stringstream(line_p);
		while (ss >> word)
		{
			if (word != "true")
				terms.push_back(word);
			if (word == "false")
				found_false = true;
		}
		if (terms.size() == 0)
			terms.push_back("true");
		word = Commatize(terms,"and");
		
		// Perform the replacement.
		if (found_false)
			ReplaceSubstring(line,"gNa("+line_substr+")","false");
		else
			ReplaceSubstring(line,"gNa("+line_substr+")",word);
		
		// Look for next gNa.
		found_false = false;
		terms.clear();
		n = line.find("gNa(");
	}
	
	// Simplify 'ors'.
	n = line.find("gNo(");
	while (n != std::string::npos)
	{
		// Get the (outer) substring enclosed by parantheses.
		line_substr = line.substr(n+4,line.size());
		line_substr = GetContentBetweenLimitChars(line_substr,'(',')');
		line_p = line_substr;

		// Load the terms in the conditional.
// 		ReplaceSubstring(line_p," ","");
// 		ReplaceSubstring(line_p,"or"," ");
// 		ReplaceSubstring(line_p,"||"," ");
		ReplaceStringsBetweenLimitStrings(line_p," ","","(",")");
		ReplaceStringsBetweenLimitStrings(line_p,"or"," ","(",")");
		ReplaceStringsBetweenLimitStrings(line_p,"||"," ","(",")");
		ss.clear();
		ss = std::stringstream(line_p);
		while (ss >> word)
		{
			if (word != "false")
				terms.push_back(word);
			if (word == "true")
				found_true = true;
		}
		if (terms.size() == 0)
			terms.push_back("false");
		word = Commatize(terms,"or");
		
		// Perform the replacement.
		if (found_true)
			ReplaceSubstring(line,"gNo("+line_substr+")","true");
		else
			ReplaceSubstring(line,"gNo("+line_substr+")",word);
		
		// Look for next gNa.
		found_true = false;
		terms.clear();
		n = line.find("gNo(");
	}
	
	return 0;
}







int Process_OutputLineSubs(ImportedSet &iset, std::string set_label, std::string &s_in)
{
	// Parameters.
	std::string line = "";
	std::stringstream ss = std::stringstream(s_in);
	
	s_in = "";
	while (ss.peek() != EOF)
	{
		line = CollectLine(ss);
		//ProcessLineSumsAndProds(line);
		ProcessLineRegular(iset, set_label, ss, line);
		//ProcessLineConditionConstructor(line);
		s_in += line;
	}
	
	return 0;
}
