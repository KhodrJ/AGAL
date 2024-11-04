#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>
#include <sstream>
#include <map>
#include <algorithm>
#include "exprtk.hpp"

#define TAB '\t'
#define SENTINEL 100000

exprtk::expression<double> expression;
exprtk::parser<double> parser;

// Auxilliary.
std::vector<std::string> logical_operators = std::vector<std::string>{"and", "nand", "nor", "not", "or", "xor", "xnor", "mand", "mor"};
std::vector<std::string> keyword_list = std::vector<std::string>{"for","if","while"};
std::vector<std::string> warning_list = std::vector<std::string>{"END_INIF","END_INFOR","END_LOOPBLOCKS"};
std::vector<std::string> coords = std::vector<std::string>{"I_kap","J_kap","K_kap"};
std::map<std::string,std::vector<std::string>> custom_import_vec;
std::map<std::string,std::vector<std::vector<std::string>>> custom_import_tab;
std::map<std::string, std::string> custom_def;

// File, kernel, and routine parameters.
std::string file_name = "";
std::string kernel_name = "Unnamed";
std::vector<std::string> kernel_includes;
std::string kernel_include_guard = "";
std::vector<std::string> kernel_args;
std::vector<std::string> kernel_args_template;
std::vector<std::string> kernel_template_args;
std::vector<std::string> kernel_template_vals;
bool started_require = false;
bool replace_mesh_identifiers = false;
std::vector<std::string> routine_args;
std::vector<std::string> routine_template_conds;
std::string routine_object = "";
std::string output_dir = "../";





/*
88888888888 .d88888b.  8888888b.   .d88888b.  
    888    d88P" "Y88b 888  "Y88b d88P" "Y88b 
    888    888     888 888    888 888     888 
    888    888     888 888    888 888     888 
    888    888     888 888    888 888     888 
    888    888     888 888    888 888     888 
    888    Y88b. .d88P 888  .d88P Y88b. .d88P 
    888     "Y88888P"  8888888P"   "Y88888P"  
*/





/*
 * 
 *  1. Implement removal of duplicate kernel arguments. Also, presets should insert required arguments.
 *  2. Option to re-use registers in an intelligent way (as a means to reduce register pressure, if possible).
 *  
 */





/*
8888888b.                                                     
888   Y88b                                                    
888    888                                                    
888   d88P 888d888 .d88b.   .d8888b .d88b.  .d8888b  .d8888b  
8888888P"  888P"  d88""88b d88P"   d8P  Y8b 88K      88K      
888        888    888  888 888     88888888 "Y8888b. "Y8888b. 
888        888    Y88..88P Y88b.   Y8b.          X88      X88 
888        888     "Y88P"   "Y8888P "Y8888   88888P'  88888P' 
*/





std::string replace_substring(std::string s, std::string subs, std::string new_subs, size_t counter_limit=std::string::npos)
{
	// Replaces substrings (subs) in a main string (s) with new substrings (new_subs).
	std::string new_string = s;
	int len = subs.length();
	size_t replacement_counter = 0;

	// Traverse string until all found instances of 'subs' are replaced.
	size_t pos_curr = 0;
	pos_curr = new_string.find(subs);
	while (pos_curr != std::string::npos && replacement_counter < counter_limit)
	{
		new_string = new_string.substr(0,pos_curr) + new_subs + new_string.substr(pos_curr+len, new_string.length());
		pos_curr = new_string.find(subs, pos_curr+new_subs.size());
		replacement_counter++;
	}

	return new_string;
}


bool string_is_number(std::string s)
{
	std::string w = replace_substring(s,"-","");
	w = replace_substring(w,".","");
	return w.size() > 0 && w.find_first_not_of("0123456789")==std::string::npos;
}


int count_substring(std::string s, std::string subs)
{
	// Replaces substrings (subs) in a main string (s) with new substrings (new_subs).
	std::string new_string = s;
	int len = subs.length();
	size_t replacement_counter = 0;

	// Traverse string until all found instances of 'subs' are replaced.
	size_t pos_curr = 0;
	pos_curr = new_string.find(subs);
	while (pos_curr != std::string::npos)
	{
		new_string = new_string.substr(0,pos_curr) + new_string.substr(pos_curr+len, new_string.length());
		pos_curr = new_string.find(subs);
		replacement_counter++;
	}

	return replacement_counter;
}


std::string tab(int inc)
{
	// Builds the appropriate indentation substring.
	std::string t = "";
	for (int k = 0; k < inc; k++)
		t = t + TAB;
	
	return t;
}


std::string add_statement(std::string output, std::string statement, int inc_curr)
{
	std::string new_output = output;
	
	new_output = new_output + tab(inc_curr) + statement + "\n";
	
	return new_output;
}


std::string do_math(std::string line)
{
	std::string new_line = line;
	std::string inner_line;
	
	size_t pos_curr_start = 0;
	size_t pos_curr_end = line.size()-1;
	pos_curr_start = line.find("^D<");
	while (pos_curr_start != std::string::npos)
	{
		pos_curr_end = line.find(">^", pos_curr_start);
		inner_line = line.substr(pos_curr_start+3, pos_curr_end-(pos_curr_start+3));
		
		
		// Do math.
		if (parser.compile(inner_line,expression))
		{
			double t = expression.value();
			char result_chars[100];
			std::sprintf(result_chars,"%17.15f",t);
			line = line.substr(0,pos_curr_start) + std::string(result_chars) + line.substr(pos_curr_start+(pos_curr_end-pos_curr_start+2), line.length());
			
			pos_curr_start = line.find("^D<");
		}
		else
			pos_curr_start = line.find("^D<", pos_curr_start+1);
	}
	//new_line = line;
	
	pos_curr_start = 0;
	pos_curr_end = line.size()-1;
	pos_curr_start = line.find("^I<");
	while (pos_curr_start != std::string::npos)
	{
		pos_curr_end = line.find(">^", pos_curr_start);
		inner_line = line.substr(pos_curr_start+3, pos_curr_end-(pos_curr_start+3));
		
		
		// Do math.
		if (parser.compile(inner_line,expression))
		{
			int t = expression.value();
			char result_chars[32];
			std::sprintf(result_chars,"%i",t);
			line = line.substr(0,pos_curr_start) + std::string(result_chars) + line.substr(pos_curr_start+(pos_curr_end-pos_curr_start+2), line.length());
			
			pos_curr_start = line.find("^I<");
		}
		else
			pos_curr_start = line.find("^I<", pos_curr_start+1);
	}
	new_line = line;
	
	return new_line;
}


std::string replace_sum_or_prod(std::string line)
{
	std::string new_line = line;
	std::string inner_line;
	
	size_t pos_curr_start = 0;
	size_t pos_curr_start_sum = 0;
	size_t pos_curr_start_prod = 0;
	size_t pos_curr_mid = 0;
	size_t pos_curr_midstart = 0;
	size_t pos_curr_end = line.size()-1;
	std::string start_term = "";
	std::string end_term = "";
	std::string op_term = "";
	pos_curr_start_sum = line.find("SUM<");
	pos_curr_start_prod = line.find("PROD<");
	pos_curr_start = std::min(pos_curr_start_sum, pos_curr_start_prod);
	while (pos_curr_start != std::string::npos)
	{
		// Is it a sum or product?
		if (pos_curr_start == pos_curr_start_sum)
		{
			start_term = "SUM<";
			end_term = ">END_SUM";
			op_term = "+";
		}
		else
		{
			start_term = "PROD<";
			end_term = ">END_PROD";
			op_term = "*";
		}
		
		
		// Find the correct 'END_SUM/END_PROD'.
		int start_counter = 1;
		int end_counter = 0;
		pos_curr_midstart = pos_curr_start;
		while (start_counter != end_counter)
		{
			pos_curr_mid = line.find(start_term, pos_curr_midstart+1);
			pos_curr_end = line.find(end_term, pos_curr_midstart+1);
			if (pos_curr_mid < pos_curr_end)
				start_counter++;
			else
				end_counter++;
			
			pos_curr_midstart = std::min(pos_curr_mid, pos_curr_end);
		}
		inner_line = line.substr(pos_curr_start+start_term.size(), pos_curr_end-(pos_curr_start+start_term.size()));
		inner_line = do_math(inner_line);
		
		
		// Retrieve loop data.
		std::stringstream ss(inner_line);
		std::string loop_index = "";
		int loop_start = 0;
		int loop_end = 1;
		int loop_increment = 1;
		ss >> loop_index >> loop_start >> loop_end >> loop_increment;
		ss.get();
		std::string remainder;
		std::getline(ss, remainder);
		
		
		// Expand the sum and adjust the line.
		std::string collector = "";
		for (int p = loop_start; p < loop_end; p += loop_increment)
		{
			if (p == loop_start)
				collector = replace_substring(remainder, std::string("<"+loop_index+">"), std::to_string(p));
			else
				collector = collector + op_term + replace_substring(remainder, std::string("<"+loop_index+">"), std::to_string(p));
		}
		line = line.substr(0,pos_curr_start) + collector + line.substr(pos_curr_start+(pos_curr_end-pos_curr_start+end_term.size()), line.length());
		
		
		// Find the next instance of 'SUM/PROD'.
		pos_curr_start_sum = line.find("SUM<");
		pos_curr_start_prod = line.find("PROD<");
		pos_curr_start = std::min(pos_curr_start_sum, pos_curr_start_prod);
	}
	new_line = line;
	
	return new_line;
}


std::string replace_sum(std::string line)
{
	std::string new_line = line;
	std::string inner_line;
	
	size_t pos_curr_start = 0;
	size_t pos_curr_mid = 0;
	size_t pos_curr_midstart = 0;
	size_t pos_curr_end = line.size()-1;
	pos_curr_start = line.find("SUM<");
	while (pos_curr_start != std::string::npos)
	{
		// Find the correct 'END_SUM'.
		int start_counter = 1;
		int end_counter = 0;
		pos_curr_midstart = pos_curr_start;
		while (start_counter != end_counter)
		{
			pos_curr_mid = line.find("SUM<", pos_curr_midstart+1);
			pos_curr_end = line.find(">END_SUM", pos_curr_midstart+1);
			if (pos_curr_mid < pos_curr_end)
				start_counter++;
			else
				end_counter++;
			
			pos_curr_midstart = std::min(pos_curr_mid, pos_curr_end);
		}
		inner_line = line.substr(pos_curr_start+4, pos_curr_end-(pos_curr_start+4));
		
		
		// Retrieve loop data.
		std::stringstream ss(inner_line);
		std::string loop_index = "";
		int loop_start = 0;
		int loop_end = 1;
		int loop_increment = 1;
		ss >> loop_index >> loop_start >> loop_end >> loop_increment;
		ss.get();
		std::string remainder;
		std::getline(ss, remainder);
		
		
		// Expand the sum and adjust the line.
		std::string collector = "";
		for (int p = loop_start; p < loop_end; p += loop_increment)
		{
			if (p == loop_start)
				collector = replace_substring(remainder, std::string("<"+loop_index+">"), std::to_string(p));
			else
				collector = collector + " + " + replace_substring(remainder, std::string("<"+loop_index+">"), std::to_string(p));
		}
		line = line.substr(0,pos_curr_start) + collector + line.substr(pos_curr_start+(pos_curr_end-pos_curr_start+8), line.length());
		
		
		// Find the next instance of 'SUM'.
		pos_curr_start = line.find("SUM<");
	}
	new_line = line;
	
	return new_line;
}


std::string replace_imports(std::string line)
{
	std::string new_line = line;
	std::string inner_line;
	
	size_t pos_curr_start = 0;
	size_t pos_curr_end = line.size()-1;
	pos_curr_start = line.find("IMP_VEC(");
	while (pos_curr_start != std::string::npos)
	{
		pos_curr_end = line.find(")END_IMP", pos_curr_start);
		inner_line = line.substr(pos_curr_start+8, pos_curr_end-(pos_curr_start+8));
		
		
		std::stringstream ss(replace_substring(inner_line,","," "));
		std::string label;
		std::string val;
		ss >> label >> val;
		
		if (string_is_number(val) && (custom_import_vec.find(label) != custom_import_vec.end()))
		{
			int i_val = std::stoi(val);
			std::string result_import = "";
			if (i_val == -1)
				result_import = std::to_string( custom_import_vec[label].size() );
			else
				result_import = custom_import_vec[label][i_val];
			line = line.substr(0,pos_curr_start) + result_import + line.substr(pos_curr_start+(pos_curr_end-pos_curr_start+8), line.length());
			pos_curr_start = line.find("IMP_VEC(");
		}
		else
			pos_curr_start = line.find("IMP_VEC(", pos_curr_start+1);
	}
	//new_line = line;
	
	pos_curr_start = 0;
	pos_curr_end = line.size()-1;
	pos_curr_start = line.find("IMP_TAB(");
	while (pos_curr_start != std::string::npos)
	{
		pos_curr_end = line.find(")END_IMP", pos_curr_start);
		inner_line = line.substr(pos_curr_start+8, pos_curr_end-(pos_curr_start+8));
		
		
		std::stringstream ss(replace_substring(inner_line,","," "));
		std::string label;
		std::string row;
		std::string val;
		ss >> label >> row >> val;
		
		if ((string_is_number(row) && string_is_number(val)) && (custom_import_tab.find(label) != custom_import_tab.end()))
		{
			int i_row = std::stoi(row);
			int i_val = std::stoi(val);
			std::string result_import = "";
			if (i_row == -1)
				result_import = std::to_string( custom_import_tab[label].size() );
			else if (i_val == -1)
				result_import = std::to_string( custom_import_tab[label][i_row].size() );
			else
			{
				if (i_val >= custom_import_tab[label][i_row].size())
					result_import = "18446744073709551615";
				else
					result_import = custom_import_tab[label][i_row][i_val];
			}
			line = line.substr(0,pos_curr_start) + result_import + line.substr(pos_curr_start+(pos_curr_end-pos_curr_start+8), line.length());
			pos_curr_start = line.find("IMP_TAB(");
		}
		else
			pos_curr_start = line.find("IMP_TAB(", pos_curr_start+1);
	}
	new_line = line;
	
	return new_line;
}


std::string commatize(std::vector<std::string> strings, std::string comma)
{
	std::string s = "";
	
	if (strings.size() > 0)
	{
		s = strings[0];
		if (strings.size() > 1)
		{
			for (int k = 1; k < strings.size(); k++)
				s = s + comma + strings[k];
		}
	}
	
	return s;
}


std::string process_regular_statement(std::string line)
{
	std::string new_line = line;
	new_line = replace_substring(new_line, "REG ", "");
	new_line = replace_substring(new_line, "    ", "");
	new_line = replace_substring(new_line, "\t", "");
	new_line = replace_substring(new_line, "\\t", "\t");
	//new_line = replace_sum_or_prod(new_line);
	//new_line = replace_prod(new_line);
	new_line = do_math(new_line);
	new_line = replace_imports(new_line);
	new_line = do_math(new_line);
	
	return new_line;
}

#include "lbm.h"





/*
8888888888                                      888    888    d8b                   
888                                             888    888    Y8P                   
888                                             888    888                          
8888888  .d88b.  888d888 88888b.d88b.   8888b.  888888 888888 888 88888b.   .d88b.  
888     d88""88b 888P"   888 "888 "88b     "88b 888    888    888 888 "88b d88P"88b 
888     888  888 888     888  888  888 .d888888 888    888    888 888  888 888  888 
888     Y88..88P 888     888  888  888 888  888 Y88b.  Y88b.  888 888  888 Y88b 888 
888      "Y88P"  888     888  888  888 "Y888888  "Y888  "Y888 888 888  888  "Y88888 
                                                                                888 
                                                                           Y8b d88P 
                                                                            "Y88P"  
*/





int FormatLineInFor(std::istream *input, std::string line, std::string *s_output);
int FormatLineInIf(std::istream *input, std::string line, std::string *s_output);


int ProcessLinesCollect(std::istream *input, std::string *s_output, std::string term_string)
{
	// Parameters.
	bool finished = false;
	int sentinel = 0;
	std::string word;
	std::string line;
	std::string collected_string = "";
	
	// Collect all lines up till the specified terminating string 'term_string'.
	while (!finished && sentinel < SENTINEL)
	{
		std::getline((*input), line);
		line = replace_substring(line, "    ", "");
		line = replace_sum_or_prod(line);
		std::stringstream ss_inner(line);
		ss_inner >> word;
		
		if (word == term_string) // e.g. END_INFOR
			finished = true;
		else
		{
			if (term_string == "END_INFOR")
			{
				std::string s = "";
				if (FormatLineInFor(input, line, &s) == 0)
					line = s;
			}
			if (term_string == "END_INIF")
			{
				std::string s = "";
				if (FormatLineInIf(input, line, &s) == 0)
					line = s;
			}
			
			// Either keep on line (removing the last '\') or add new-line character.
			if (line[line.size()-1] == '\\')
				line = std::string(line.c_str(), line.size()-1) + " ";
			else
			{
				if (line[line.size()-1] != '\n' && line != "")
					line = line + "\n";
			}

			collected_string = collected_string + line;
		}
		
		sentinel++;
	}
	
	(*s_output) = (*s_output) + collected_string;
	
	return 0;
}


// For collection and replacement of code presets.
int RetrieveInput(std::istream *input, std::string line, std::string *s_output)
{
	std::string word = "";
	std::stringstream ss(line);
	
	
	// Proceed if non-empty line.
	if (ss >> word)
	{
		bool regular_statement = true;
		
		
		if (word == "IMPORT")
		{
			regular_statement = false;
			ss >> word;
			
			std::string filename;
			std::string label;
			std::string inner_line;
			ss >> filename >> label;
			
			// Load file containing data to be imported and process.
			std::ifstream file = std::ifstream("import/"+filename);
			if (word == "VECTOR")
			{
				std::vector<std::string> imported_vector;
				while (file >> word && word != "END")
					imported_vector.push_back(word);
				//if (custom_import_vec.find(label) != custom_import_vec.end())
				//	std::cout << "[E] Warning: vector with same label already imported, overwriting..." << std::endl;
				custom_import_vec[label] = imported_vector;
				std::cout << "[-] Imported VECTOR (" << label << ")..." << std::endl;
			}
			else if (word == "TABLE")
			{
				std::vector<std::vector<std::string>> imported_tab;
				for (inner_line; std::getline(file, inner_line);)
				{
					if (inner_line.find("END") == std::string::npos)
					{
						std::stringstream inner_ss(inner_line);
						std::vector<std::string> imported_row_vector;
						while (inner_ss >> word)
							imported_row_vector.push_back(word);
						imported_tab.push_back(imported_row_vector);
					}
					else
						file.seekg(0, std::ios::end);
				}
				//if (custom_import_tab.find(label) != custom_import_tab.end())
				//	std::cout << "[E] Warning: table with same label already imported, overwriting..." << std::endl;
				custom_import_tab[label] = imported_tab;
				std::cout << "[-] Imported TABLE (" << label << ")..." << std::endl;
			}
			else
				std::cout << "[E] Invalid import type..." << std::endl;
			file.close();
		}
		
		
		if (word == "END_FILE")
		{
			regular_statement = false;
			
			(*input).seekg(0, std::ios::end);
		}
		
		
		if (regular_statement)
		{
			// First attempt at replacing imported labels. This is done again at the end after for-loop expansion and math evaluations.
			line = replace_imports(line);
			
			if (word[0] != '#')
				(*s_output) = (*s_output) + line + "\n";
		}
	}
	
	return 0;
}


// For collection and replacement of code presets.
int FormatLinePre(std::istream *input, std::string line, std::string *s_output)
{
	std::string word = "";
	std::stringstream ss(line);
	
	
	// Proceed if non-empty line.
	if (ss >> word)
	{
		bool regular_statement = true;
		
		if (word == "LOOPBLOCKS")
		{
			regular_statement = false;
			
			// Process arguments.
			int n_args = 0;
			int n_reqs = 0;
			std::string arg_k;
			std::string cond = "";
			std::string sub_cond = "";
			std::vector<std::string> sub_reqs;
			std::vector<std::string> args_k;
			ss >> n_args;
			for (int k = 0; k < n_args; k++)
			{
				ss >> arg_k;
				args_k.push_back(arg_k);
				
				if (arg_k == "CONDITION")
				{
					ss >> word;
					cond = " && " + word;
				}
				else if (arg_k == "REQUIRING")
				{
					ss >> n_reqs;
					for (int j = 0; j < n_reqs; j++)
					{
						ss >> word;
						sub_reqs.push_back(word);
					}
				}
				else
				{
					std::cout << "Error processing LOOPBLOCKS: Undefined condition (" << arg_k << ")..." << std::endl;
				}
			}
			
			std::string s = "\
			int kap = blockIdx.x*M_LBLOCK + threadIdx.x;\n\
			<\n\
			s_ID_cblock[threadIdx.x] = -1;\n\
			OUTIF (threadIdx.x < M_LBLOCK) and (kap < n_ids_idev_L)\n\
			s_ID_cblock[threadIdx.x] = id_set_idev_L[kap];\n\
			END_OUTIF\n\
			__syncthreads();\n\
			<\n\
			// Loop over block Ids.\n\
			OUTFOR k 1   0 M_LBLOCK 1\n\
			i_kap_b = s_ID_cblock[k];\n";
			//__syncthreads();\n";
			
			if (sub_reqs.size() > 0)
			{
				s = s + "\
				<\n\
				// This part is included if n>0 only.\n\
				OUTIF i_kap_b > -1\n";
				
				for (int j = 0; j < sub_reqs.size(); j++)
					s = s + sub_reqs[j] + ";\n";
					
				s = s + "\
				END_OUTIF\n";
			}
			
			s = s + "\
			<\n\
			// Latter condition is added only if n>0.\n\
			OUTIF i_kap_b > -1" + cond + "\n\
			";
			
			s = replace_substring(s,"\t","");
			(*s_output) = (*s_output) + s;
		}
		
		if (word == "END_LOOPBLOCKS")
		{
			regular_statement = false;
			
			
			(*s_output) = (*s_output) + replace_substring(std::string("\
			END_OUTFOR 1\n\
			END_OUTIF\n\
			"),"\t","");
		}
		
		if (regular_statement)
		{
			(*s_output) = (*s_output) + line + "\n";
		}
	}
	
	return 0;
}


// For collection and processing of in-if statements within collected lines from input file.
int FormatLineInIf(std::istream *input, std::string line, std::string *s_output)
{
	std::string word = "";
	std::stringstream ss(line);
	
	
	// Proceed if non-empty line.
	if (ss >> word)
	{
		bool regular_statement = true;
		
		// In-if conditions (i.e., within code generator):
		// - Only process block of lines up till END_INIF if condition is satisfied.
		if (word == "INIF")
		{
			regular_statement = false;
			
			// Process the condition.
			std::string cond = "";
			cond = line.substr(line.find("INIF")+5,line.size());
			if (parser.compile(cond,expression))
			{
				int t = expression.value();
				
				// Collect lines between FOR and ENDFOR in one string to be looped.
				std::string collected_string = "";
				ProcessLinesCollect(input, &collected_string, "END_INIF");
				
				// Divide the collected string if an INELSE condition exists.
				size_t else_pos = 0;
				std::string collected_string_if = "";
				std::string collected_string_else = "";
				else_pos = collected_string.find("INELSE");
				if (else_pos != std::string::npos)
				{
					collected_string_if = collected_string.substr(0, else_pos);
					collected_string_else = collected_string.substr(else_pos+6, collected_string.size());
				}
				else
					collected_string_if = collected_string;
				
				if (t)
					(*s_output) = (*s_output) + collected_string_if;
				else
					(*s_output) = (*s_output) + collected_string_else;
			}
			else
			{
				(*s_output) = (*s_output) + line + "\n";
				
				std::cout << "Failed to compile: " << cond << " (original line: " << line << ")..." << std::endl;
			}
		}
		
		if (regular_statement)
		{
			if (line[line.size()-1] != '\\')
				line = line + "\n";
			else
				line = std::string(line.c_str(), line.size()-1);
			(*s_output) = (*s_output) + line;
			
			return 1;
		}
	}
	
	return 0;
}


// For collection and expansion of lines from the input file.
int FormatLineInFor(std::istream *input, std::string line, std::string *s_output)
{
	std::string word = "";
	std::stringstream ss(line);
	
	
	// Proceed if non-empty line.
	if (ss >> word)
	{
		bool regular_statement = true;
		
		// In-for loop (i.e., within code generator):
		// - Collect all lines between INFOR and END_INFOR, then iterate over loop range while replacing indices.
		if (word == "INFOR")
		{
			regular_statement = false;
			line = do_math(line);
			ss = std::stringstream(line);
			ss >> word;
			
			// Parameters.
			std::string loop_index_subs_string = "";
			int loop_index_subs_len = 1;
			int tmp_data_int[3];
			std::vector<std::string> loop_index_subs;
			std::vector<int> loop_data;
			std::vector<int> loop_vols;
			std::string collected_string = "";
			std::string inter_string = "";
			std::string final_string = "";
			int range_vol = 1;
			
			// Get loop indices.
			ss >> loop_index_subs_string;
			ss >> loop_index_subs_len;
			for (int k = 0; k < loop_index_subs_string.size()/loop_index_subs_len; k++)
				loop_index_subs.push_back(std::string(&loop_index_subs_string[k*loop_index_subs_len],loop_index_subs_len));
			
			// Get loop ranges and increments.
			for (int k = 0; k < loop_index_subs.size(); k++)
			{
				ss >> tmp_data_int[0] >> tmp_data_int[1] >> tmp_data_int[2];
				tmp_data_int[1]--; // Exclusive upper bound.
				
				loop_data.push_back(tmp_data_int[0]); // Range (lower).
				loop_data.push_back(tmp_data_int[1]); // Range (upper).
				loop_data.push_back(tmp_data_int[2]); // Increment.
				
				int vol_k = (tmp_data_int[1]-tmp_data_int[0]+1);
				loop_vols.push_back(vol_k);
				range_vol *= vol_k;
			}
			
			// Collect lines between FOR and ENDFOR in one string to be looped.
			ProcessLinesCollect(input, &collected_string, "END_INFOR");
			
			// Loop and replace indices in statements.
			int *kaps = new int[loop_index_subs.size()];
			int *Is = new int[loop_index_subs.size()];
			for (int K = 0; K < range_vol; K++)
			{
				// Intermediate variables.
				kaps[0] = K;
				for (int k = 1; k < loop_index_subs.size(); k++)
					kaps[k] = kaps[k-1]/loop_vols[k-1];
				
				// Calculate loop indices.
				for (int k = 0; k < loop_index_subs.size(); k++)
					Is[k] = kaps[k]%loop_vols[k] + loop_data[0+k*3];
				
				// Check if current indices are consistent with increments.
				bool proceed_indices = true;
				for (int k = 0; k < loop_index_subs.size(); k++)
				{
					if ((Is[k]+loop_data[0+k*3])%loop_data[2+k*3] != 0)
						proceed_indices = false;
				}
				
				// Update final string.
				if (proceed_indices)
				{
					inter_string = collected_string;
					for (int k = 0; k < loop_index_subs.size(); k++)
					{
						std::string subs = "<" + loop_index_subs[k] + ">";
						inter_string = replace_substring(inter_string, subs, std::to_string(Is[k]));
					}
					final_string = final_string + inter_string;
				}
			}
			delete[] kaps;
			delete[] Is;
			
			// Last character should be a new character.
			if (final_string[final_string.size()-1] != '\n')
				final_string = final_string + "\n";
			
			(*s_output) = (*s_output) + final_string;
		}
		
		if (regular_statement)
		{
			line = replace_sum_or_prod(line);
			
			if (line[line.size()-1] != '\\')
				line = line + "\n";
			else
				line = std::string(line.c_str(), line.size()-1);
			(*s_output) = (*s_output) + line;
			
			return 1;
		}
	}
	
	
	return 0;
}


// For processing of the collected string.
int FormatLinePostprocess(int *indent_tracker, std::string line, std::string *s_output)
{
	std::string word = "";
	std::stringstream ss(line);
	
	
	// Proceed if non-empty line.
	if (ss >> word)
	{
		bool regular_statement = true;
		if (word != "DEFINE")
		{
			for (auto it = custom_def.begin(); it != custom_def.end(); ++it)
				line = replace_substring(line, it->first, it->second);
		}
		line = process_regular_statement(line);
		
		
		if (word == "<")
		{
			regular_statement = false;
			
			
			(*s_output) = (*s_output) + "\n";
		}
		
		
		/*
		if (word == "IMPORT")
		{
			regular_statement = false;
			
			ss >> word;
			if (word == "VECTOR")
			{
				std::string filename;
				std::string label;
				ss >> filename >> label;
				
				std::vector<std::string> imported_vector;
				std::ifstream file = std::ifstream("import/"+filename);
				while (file >> word)
					imported_vector.push_back(word);
				file.close();
				
				if (custom_import_vec.find(label) != custom_import_vec.end())
					std::cout << "[E] Warning: vector with same label already imported, overwriting..." << std::endl;
				custom_import_vec[label] = imported_vector;
			}
		}
		*/
		
		
		if (word == "DEFINE")
		{
			regular_statement = false;
			int mode = 0;
			
			std::string var_name = "";
			std::string tmp = "";
			ss >> var_name;
			if (var_name == "DEF_ADD")
			{
				mode = 1;
				ss >> var_name;
			}
			if (var_name == "DEF_PUSH")
			{
				mode = 2;
				ss >> tmp >> var_name;
			}
			
			std::string inner_line = line.substr(line.find(var_name)+var_name.size()+1, line.size());
			inner_line = do_math(inner_line);
			inner_line = replace_sum_or_prod(inner_line);
			
			if (mode == 0)
				custom_def[var_name] = inner_line;
			if (mode == 1)
				custom_def[var_name] = custom_def[var_name] + inner_line;
			if (mode == 2)
			{
				if (custom_def[var_name] == "")
					custom_def[var_name] = inner_line;
				else
					custom_def[var_name] = custom_def[var_name] + tmp + inner_line;
			}
		}
		
		
		if (word == "UNDEFINE")
		{
			regular_statement = false;
			
			std::string var_name = "";
			ss >> var_name;
			custom_def.erase(var_name);
		}
		
		
		if (word == "NAME")
		{
			regular_statement = false;
			
			std::string new_name = "";
			ss >> new_name;
			kernel_name = new_name;
		}
		
		
		if (word == "NAME_FILE")
		{
			regular_statement = false;
			
			
			std::string new_name = "";
			ss >> new_name;
			file_name = new_name;
		}
		
		
		if (word == "INCLUDE")
		{
			regular_statement = false;
			
			std::string var = "";
			ss >> var;
			kernel_includes.push_back(var);
		}
		
		
		if (word == "INCLUDE_GUARD")
		{
			regular_statement = false;
			
			std::string var = line.substr(14, line.size());
			kernel_include_guard = var;
		}
		
		
		if (word == "KERNEL_REPLACE_MESH_VARS")
		{
			regular_statement = false;
			
			replace_mesh_identifiers = true;
		}
		
		
		if (word == "KERNEL_REQUIRE")
		{
			regular_statement = false;
			if (started_require = false)
				started_require = true;
			
			
			//size_t pos = line.find("TEMPLATE");
			std::string var_identifier = line.substr(15,line.size());
			//std::string var_template_vals = line.substr(pos+9,line.size());
			
			
			// Clean string.
			std::string word = "";
			std::vector<std::string> words_kernel;
			std::vector<std::string> words_template;
			std::vector<std::string> words_routine;
			std::stringstream ss(var_identifier);
			while (ss >> word)
			{
				if (word == "ROUTINE")
				{
					int N_routine_args = std::max((int)kernel_template_vals.size(),1);
// 					ss >> N_routine_args;
// 					if (N_routine_args < kernel_template_vals.size())
// 						N_routine_args = 1;
// 					if (N_routine_args > kernel_template_vals.size())
// 					{
// 						std::cout << "[E] Warning: " << N_routine_args << " specified, although only " << kernel_template_vals.size() << " sets of template values exist. Reading " << std::max((int)kernel_template_vals.size(),1) << " values instead..." << std::endl;
// 						N_routine_args = std::max((int)kernel_template_vals.size(),1);
// 					}
					
					// If TEMPLATE is specified, N_{T_VALS} arguments must be specified.
					for (int k = 0; k < N_routine_args; k++)
					{
						if (ss >> word)
							words_template.push_back(word);
						else
							words_template.push_back(words_template[words_template.size()-1]);
					}
				}
				else
				{
					//word = word.substr(0, word.find("["));
					words_kernel.push_back(word);
				}
			}
			kernel_args.push_back( commatize(words_kernel," ") );
			
			
			// Add template versions.
			// NOTE: Assumes that template values have been selected, DO NOT add more T_VALS after REQUIRE(s).
			word = "";
			ss = std::stringstream( commatize(words_template," ") );
			for (int k = 0; k < std::max((int)kernel_template_vals.size(),1); k++)
			{
				if (words_template.size() == 0)
					word = words_kernel[words_kernel.size()-1];
				else if (words_template.size() == 1)
					word = words_template[0];
				else
					ss >> word;
				
				kernel_args_template.push_back(word);
			}
		}
		
		
		if (word == "ROUTINE_REQUIRE")
		{
			regular_statement = false;
			
			
			std::string word;
			std::vector<std::string> words;
			std::string var_identifier = line.substr(16,line.size());
			//while (ss >> word)
			//	words.push_back(word);
			routine_args.push_back( var_identifier );
		}
		
		
		if (word == "ROUTINE_COND")
		{
			regular_statement = false;
			
			
			std::string var = line.substr(13, line.size());
			if (routine_template_conds.size() < std::max((int)kernel_template_vals.size(),1))
				routine_template_conds.push_back(var);
		}
		
		
		if (word == "ROUTINE_OBJECT")
		{
			regular_statement = false;
			
			
			ss >> word;
			routine_object = word + "::";
		}
		
		
		if (word == "TEMPLATE")
		{
			regular_statement = false;
			
			std::string var = line.substr(9,line.size());
			kernel_template_args.push_back(var);
		}
		
		
		if (word == "TEMPLATE_VALS")
		{
			regular_statement = false;
			if (started_require == true)
				std::cout << "[E] WARNING: Do not insert more T_VALS after REQUIRE, only before..." << std::endl; 
			
			std::string var = line.substr(14,line.size());
			kernel_template_vals.push_back(var);
		}
		
		
		if (word == "OUTPUT_DIR")
		{
			regular_statement = false;
			
			
			ss >> word;
			output_dir = word;
		}
		
		
		// Starts an outer for-loop (not inherent to the generator, loops in the kernel).
		if (word == "OUTFOR")
		{
			regular_statement = false;
			
			
			// Parameters.
			std::string loop_index_subs_string = "";
			int loop_index_subs_len = 1;
			std::string tmp_data_int[3];
			std::vector<std::string> loop_index_subs;
			std::vector<std::string> loop_data;
			
			
			// Get loop indices.
			ss >> loop_index_subs_string;
			ss >> loop_index_subs_len;
			for (int k = 0; k < loop_index_subs_string.size()/loop_index_subs_len; k++)
				loop_index_subs.push_back(std::string(&loop_index_subs_string[k*loop_index_subs_len],loop_index_subs_len));
			
			
			// Get loop ranges and increments.
			for (int k = 0; k < loop_index_subs.size(); k++)
			{
				ss >> tmp_data_int[0] >> tmp_data_int[1] >> tmp_data_int[2];
				loop_data.push_back(tmp_data_int[0]); // Range (lower).
				loop_data.push_back(tmp_data_int[1]); // Range (upper).
				loop_data.push_back(tmp_data_int[2]); // Increment.
			}
			
			
			// Build the C++ loop.
			for (int k = loop_index_subs.size()-1; k >= 0; k--)
			{
				std::string opening_bracket = tab((*indent_tracker)) + "for (int " + loop_index_subs[k] + " = " + loop_data[0+k*3] + "; " + loop_index_subs[k] + " < " + loop_data[1+k*3] + "; " + loop_index_subs[k] + " += " + loop_data[2+k*3] + ")\n" + tab((*indent_tracker)) + "{\n";
				(*indent_tracker) += 1;
				
				(*s_output) = (*s_output) + opening_bracket;
			}
		}
		
		
		// Closes outer for-loops.
		if (word == "END_OUTFOR")
		{
			regular_statement = false;
			
			
			int no_closes = 1;
			ss >> no_closes;
			
			for (int k = 0; k < no_closes; k++)
			{
				(*indent_tracker)--;
				std::string closing_bracket = tab((*indent_tracker)) + "}" + "\n";
				
				(*s_output) = (*s_output) + closing_bracket;
			}
		}
		
		
		// Starts an outer if-statement.
		if (word == "OUTIF")
		{
			regular_statement = false;
			
			
			std::string cond = replace_substring( line.substr(6, line.size()), " ", "" );
			
			cond = replace_substring( line.substr(6, line.size()), " and ", "and" );
			(*s_output) = (*s_output) + tab((*indent_tracker)) + "if (" + cond + ")\n";
			(*s_output) = (*s_output) + tab((*indent_tracker)) + "{" + "\n";
			(*indent_tracker)++;
		}
		
		
		// Starts an outer if-statement.
		if (word == "OUTIF_S")
		{
			regular_statement = false;
			
			
			std::string cond = replace_substring( line.substr(8, line.size()), " ", "" );
			
			cond = replace_substring( line.substr(8, line.size()), " and ", "and" );
			(*s_output) = (*s_output) + tab((*indent_tracker)) + "if (" + cond + ")\n";
			(*indent_tracker)++;
		}
		
		
		// Writes an outer else-statement.
		if (word == "OUTELSE")
		{
			regular_statement = false;
			
			
			(*indent_tracker)--;
			(*s_output) = (*s_output) + tab((*indent_tracker)) + "}" + "\n";
			(*s_output) = (*s_output) + tab((*indent_tracker)) + "else" + "\n";
			(*s_output) = (*s_output) + tab((*indent_tracker)) + "{" + "\n";
			(*indent_tracker)++;
		}
		
		
		// Closes outer if-statement.
		if (word == "END_OUTIF")
		{
			regular_statement = false;
		
			
			(*indent_tracker)--;
			(*s_output) = (*s_output) + tab((*indent_tracker)) + "}" + "\n";
		}
		
		
		// Closes outer if-statement.
		if (word == "END_OUTIF_S")
		{
			regular_statement = false;
		
			
			(*indent_tracker)--;
		}
		
		
		// Warnings.
		for (int k = 0; k < warning_list.size(); k++)
		{
			if (word == warning_list[k])
			{
				regular_statement = false;
				std::cout << "Warning, extra \"" << warning_list[k] << "\" detected..." << std::endl;
			}
		}
		
		
		// Process regular statements.
		if (regular_statement)
		{
			// Perform processing routine.
			//line = process_regular_statement(line);
			
			// Adjust end-of-line character.
			if (line[line.size()-1] != '\\')
				line = line + "\n";
			else
				line = std::string(line.c_str(), line.size()-1);
			
			// Add indendation.
			line = tab((*indent_tracker)) + line;
			
			// Write to collector.
			(*s_output) = (*s_output) + line;
			
			return 1;
		}
	}
	
	
	return 0;
}





/*
888     888                 d8b          888      888                   
888     888                 Y8P          888      888                   
888     888                              888      888                   
Y88b   d88P 8888b.  888d888 888  8888b.  88888b.  888  .d88b.  .d8888b  
 Y88b d88P     "88b 888P"   888     "88b 888 "88b 888 d8P  Y8b 88K      
  Y88o88P  .d888888 888     888 .d888888 888  888 888 88888888 "Y8888b. 
   Y888P   888  888 888     888 888  888 888 d88P 888 Y8b.          X88 
    Y8P    "Y888888 888     888 "Y888888 88888P"  888  "Y8888   88888P' 
*/





int UpdateRegMap()
{
	// Traverse the input starting from current position. For loops are traversed twice.
	
	return 0;
}

int ReplaceVariables(std::istream *input, std::string *s_output)
{
	std::string word;
	std::string line;
	std::ofstream log = std::ofstream("log.txt");
	std::vector<std::string> var_del;
	std::vector<std::string> var_list;
	std::map<std::string, std::string> var_map;
	
	
	// Fill up var_del.
	for (int k = 255; k >= 0; k--)
		var_del.push_back("R"+std::to_string(k));
	
	
	// Traverse output and map variable identifiers (now the input of this routine).
	for (line; std::getline((*input), line);)
	{
		std::stringstream ss(line);
		
		// Only process if non-empty line.
		if (ss >> word)
		{
			// Add a new variable, record its type and identifier. Write definition to output as is.
			if (word == "REGDEF")
			{
				std::string inner_word = "";
				std::string inner_line = line.substr(0,line.find("="));
				std::stringstream inner_ss(inner_line);
				std::string identifer = "";
				while (inner_ss >> inner_word)
					identifer = inner_word;
				identifer = replace_substring(identifer.substr(0,identifer.find("[")), " ", "");\
				
				
				var_list.push_back(identifer);
				(*s_output) = (*s_output) + replace_substring(line, "REGDEF ", "") + "\n";
			}
			else // Process the line.
			{
				//std::vector<std::string> 
			}
		}
		else
			std::cout << std::endl;
	}
	
	return 0;
}





/*
888    d8P                                     888 
888   d8P                                      888 
888  d8P                                       888 
888d88K      .d88b.  888d888 88888b.   .d88b.  888 
8888888b    d8P  Y8b 888P"   888 "88b d8P  Y8b 888 
888  Y88b   88888888 888     888  888 88888888 888 
888   Y88b  Y8b.     888     888  888 Y8b.     888 
888    Y88b  "Y8888  888     888  888  "Y8888  888 
*/





int MakeHeader(std::string *s_output)
{
	std::string kernel_header = "";
	
	// Template parameters.
	if (kernel_template_args.size() > 0)
		kernel_header = kernel_header + "template <" + commatize(kernel_template_args, ", ") + ">\n";
	
	// Kernel name.
	kernel_header = kernel_header + "__global__\nvoid Cu_" + kernel_name + "\n";
	
	// Kernel arguments.
	kernel_header = kernel_header + "(\n";
	kernel_header = kernel_header + "\t" + commatize(kernel_args, ", ") + "\n";
	kernel_header = kernel_header + ")\n{\n";
	
	// Update output string.
	(*s_output) = kernel_header + (*s_output) + "}\n";
	
	return 0;
}

// TODO: Make routine arguments input-able, make kernel launch parameters input-able.
int MakeRoutine(std::string *s_output)
{
	std::string routine = "";
	for (int k = 0; k < kernel_includes.size(); k++)
		routine = routine + "#include " + kernel_includes[k] + "\n";
	routine = routine + "\n";
	if (kernel_include_guard != "")
		routine = routine + "#if (" + kernel_include_guard +")\n\n";
	(*s_output) = routine + (*s_output);
	
	routine = "\n";
	routine = routine + "int " + routine_object + "S_" + kernel_name + "(" + commatize(routine_args,", ") + ")\n{\n";
	
	
	// For each set of template values, make a new condition.
	int N_temp_vals = std::max((int)kernel_template_vals.size(),1);
	for (int k = 0; k < N_temp_vals; k++)
	{
		routine = routine + "\tif (";
		routine = routine + routine_template_conds[k];
		routine = routine + ")\n\t{\n";
		routine = routine + "\t\t" + "Cu_" + kernel_name;
		
		
		if (kernel_template_vals.size() > 0)
		{
			std::string template_vals_k = kernel_template_vals[k];
			std::stringstream ss(template_vals_k);
			std::string word = "";
			std::vector<std::string> words;
			for (int j = 0; j < kernel_template_args.size(); j++)
			{
				ss >> word;
				words.push_back(word);
			}
			routine = routine + "<" + commatize(words,",") + ">";
		}
		
		
		// Kernel requirements.
		int N_args = kernel_args.size();
		std::vector<std::string> kernel_args_identifiers;
		for (int j = 0; j < kernel_args.size(); j++)
		{
			std::string word = "";
			std::string identifier = "";
			std::stringstream ss(kernel_args_template[k+N_temp_vals*j]);
			
			
			while (ss >> word)
				identifier = word;
			//identifier = replace_substring(identifier, "*", "");
			//identifier = identifier.substr(0, identifier.find("["));
			kernel_args_identifiers.push_back(identifier);
		}
		
		
		routine = routine + "<<<(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK,M_TBLOCK,0,mesh->streams[i_dev]>>>(" + commatize(kernel_args_identifiers, ", ") + ");\n";
		
		routine = routine + "\t}\n";
	}
	
	
	routine = routine + "\n\treturn 0;\n}\n";
	if (kernel_include_guard != "")
		routine = routine + "\n#endif";
	
		
	(*s_output) = (*s_output) + routine;
	
	return 0;
}

int MakeTestFile(std::ostream *output_test, std::string *s_output)
{
	// File header.
	(*output_test) << "#include <iostream>\n";
	(*output_test) << "#define Nbx 4\n";
	(*output_test) << "#define Nqx 2\n";
	(*output_test) << "#define M_TBLOCK 64\n";
	(*output_test) << "#define N_Pf(x) (x)\n";
	(*output_test) << "typedef double ufloat_t;\n\n";
	(*output_test) << (*s_output) << "\n";
	(*output_test) << "int main(int argc, char *argv[])\n{\n";
	
	// Kernel requirements.
	std::vector<std::string> kernel_args_identifiers;
	for (int k = 0; k < kernel_args.size(); k++)
	{
		if (kernel_args[k].find("=") == std::string::npos)
			(*output_test) << "\t" << kernel_args[k] << " = 0;\n";
		else
			(*output_test) << "\t" << kernel_args[k] << ";\n";
		
		std::string word = "";
		std::string identifier = "";
		std::stringstream ss(kernel_args[k]);
		while (ss >> word)
			identifier = word;
		identifier = replace_substring(identifier, "*", "");
		identifier = identifier.substr(0, identifier.find("["));
		
		kernel_args_identifiers.push_back(identifier);
	}
	
	// Kernel call.
	(*output_test) << "\n\t// NOTE: This routine will (likely) not run, this file is just to check whether or not the kernel compiles.\n";
	(*output_test) << "\t" << "Cu_" + kernel_name << "<<<1,128>>>(" << commatize(kernel_args_identifiers, ", ") << ");\n";
	
	// Finish.
	(*output_test) << "\treturn 0;\n}\n";
	
	return 0;
}





/*
888b     d888          d8b          
8888b   d8888          Y8P          
88888b.d88888                       
888Y88888P888  8888b.  888 88888b.  
888 Y888P 888     "88b 888 888 "88b 
888  Y8P  888 .d888888 888 888  888 
888   "   888 888  888 888 888  888 
888       888 "Y888888 888 888  888 
*/





int main(int argc, char *argv[])
{
	// Parameters.
		// General.
	std::string word = "";
	std::string line = "";
	std::string collected_string = "";
	int indent_tracker = 1;
		// Output.
	std::string s_output = "";
	std::vector<std::string> var_del;
	std::map<std::string, std::string> var_spec;
	std::map<std::string, std::string> var_map;
	
	
	// Fill up var_del.
	for (int k = 255; k >= 0; k--)
		var_del.push_back("R"+std::to_string(k));
	
	
	if (argc == 1)
	{
		std::cout << "Please provide the name for the input file." << std::endl;
		return 1;
	}
	
	
	// Load input file.
	std::ifstream input = std::ifstream("./input/" + std::string(argv[1]));
	
	
	if (input)
	{
		for (int K = 0; K < LBM_Variables_Preprocess.size(); K++)
		{
			// Reset vectors and kernel parameters.
			custom_import_vec.clear();
			custom_import_tab.clear();
			custom_def.clear();
			kernel_name = "Unnamed";
			kernel_includes.clear();
			kernel_include_guard = "";
			kernel_args.clear();
			kernel_args_template.clear();
			kernel_template_args.clear();
			kernel_template_vals.clear();
			started_require = false;
			replace_mesh_identifiers = false;
			routine_args.clear();
			routine_template_conds.clear();
			routine_object = "";
			output_dir = "../";
			file_name = replace_substring(replace_substring(std::string(argv[1]),"input_",""),".agc","");
			
			
			// Collect lines from input file.
			std::cout << "[-] Collecting lines from input..." << std::endl;
			collected_string = "";
			for (line; std::getline(input, line);)
				RetrieveInput(&input, line, &collected_string);
			s_output = replace_substring(collected_string, "\\\n", " ");
			
			
			// Perform preprocessing replacements.
			std::cout << "[-] Performing preprocessing replacements..." << std::endl;
			collected_string = "";
			std::istringstream input_pre(s_output);
			for (line; std::getline(input_pre, line);)
				FormatLinePre(&input_pre, line, &collected_string);
			s_output = collected_string;
			
			
			// Do all solver-dependent variable replacements (preprocessing).
			std::cout << "[-] Performing solver-dependent variable preprocessing replacements..." << std::endl;
			for (auto it = LBM_Variables_Preprocess[K].begin(); it != LBM_Variables_Preprocess[K].end(); ++it)
				s_output = replace_substring(s_output, it->first, it->second);
			
			
			// Go back and process all in-for loops.
			std::cout << "[-] Expanding in-for loops..." << std::endl;
			collected_string = "";
			std::istringstream input_infor(s_output);
			for (line; std::getline(input_infor, line);)
				FormatLineInFor(&input_infor, line, &collected_string);
			s_output = collected_string;
			
			
			// Go back and expand sums/products, do math (Round I).
			std::cout << "[-] Expanding sums/products, doing math..." << std::endl;
			collected_string = "";
			std::istringstream input_math_I(s_output);
			for (line; std::getline(input_math_I, line);)
			{
				if (line.size() > 0)
				{
					line = replace_imports(line);
					line = replace_sum_or_prod(line);
					line = do_math(line);
					collected_string = collected_string + line + "\n";
				}
			}
			s_output = collected_string;
			
			
			// Do all solver-dependent variable replacements (postprocessing).
			std::cout << "[-] Performing solver-dependent variable postprocessing replacements..." << std::endl;
			collected_string = "";
			int iter_counter = 0;
			while (strcmp(s_output.c_str(), collected_string.c_str()) != 0)
			{
				collected_string = s_output;
				for (auto it = LBM_Variables_Postprocess[K].begin(); it != LBM_Variables_Postprocess[K].end(); ++it)
					s_output = replace_substring(s_output, it->first, it->second);
				
				std::cout << "    Iteration: " << iter_counter++ << std::endl;
			}
			
			
			// Go back and expand sums/products, do math (Round II).
			std::cout << "[-] Expanding sums/products, doing math..." << std::endl;
			collected_string = "";
			std::istringstream input_math_II(s_output);
			for (line; std::getline(input_math_II, line);)
			{
				if (line.size() > 0)
				{
					line = replace_imports(line);
					line = replace_sum_or_prod(line);
					line = do_math(line);
					collected_string = collected_string + line + "\n";
				}
			}
			s_output = collected_string;
			
			
			// Go back and process all in-if statements.
			std::cout << "[-] Process in-if statements..." << std::endl;
			collected_string = "";
			std::istringstream input_inif(s_output);
			for (line; std::getline(input_inif, line);)
				FormatLineInIf(&input_inif, line, &collected_string);
			s_output = FormatLinePost_LBM(collected_string);
			
			
			// Now perform postprocessing and interpret lines.
			std::cout << "[-] Performing general postprocessing replacements..." << std::endl;
			collected_string = "";
			std::istringstream input_post(s_output);
			for (line; std::getline(input_post, line);)
				FormatLinePostprocess(&indent_tracker, line, &collected_string);
			s_output = collected_string;
			
			
			// Clean up by simplifying 1.0/0.0s and removing redundant operators.
			
			
			
			// Variable collection.
			//collected_string = "";
			//std::istringstream input_var(s_output);
			//ReplaceVariables(&input_var, &collected_string);
			//s_output = collected_string;
			
			
			// TODO: Fix up the closing of open brackets in separate routine.
			if (indent_tracker < 1)
				std::cout << "Careful, more brackets have been closed than opened." << std::endl;
			if (indent_tracker > 1)
				std::cout << "Careful, more brackets have been opened than closed." << std::endl;
			
			
			// Create output file.
			//std::ofstream output = std::ofstream("./output/" + file_name + "_" + LBM_Variables_Preprocess[K]["LBM_name"] + ".cu");
			std::ofstream output = std::ofstream(output_dir + file_name + "_" + LBM_Variables_Preprocess[K]["LBM_name"] + ".cu");
			std::ofstream output_test = std::ofstream("./output/tests/" + file_name + "_" + LBM_Variables_Preprocess[K]["LBM_name"] + ".cu");
			
			
			// Write to output file.
			std::cout << "[-] Writing to output file..." << std::endl;
			MakeHeader(&s_output);
			MakeTestFile(&output_test, &s_output);
			MakeRoutine(&s_output);
			output << s_output;
			//for (auto it = var_map.begin(); it != var_map.end(); ++it)
			//	std::cout << it->first << " " << it->second << std::endl;
			
			
			// Close output file and reset input file.
			output.close();
			output_test.close();
			input.clear();
			input.seekg(0, std::ios_base::beg);
		}
	}
	else
	{
		std::cout << "Could not open input file..." << std::endl;
	}
	
	
	// Close input file.
	input.close();
	std::cout << "Finished." << std::endl;
	
	
	return 0;
}
