#include "gen.h"

int ProcessLineGetRoutineParameters(std::string &line,
	std::string &r_name,
	std::string &r_object_name,
	std::vector<std::string> &r_params,
	std::vector<std::string> &r_includes,
	std::vector<std::string> &r_kernel_params,
	std::vector<std::vector<std::string>> &r_kernel_inputs,
	std::string &r_template_params,
	std::vector<std::string> &r_template_vals,
	std::vector<std::string> &r_template_args,
	std::string &r_nblocks,
	std::string &r_blocksize
)
{
	// Parameters.
	int n = 0;
	int index_size = 1;
	std::string word = "";
	std::string word2 = "";
	std::vector<std::string> words3;
	std::string last_word = "";
	std::stringstream ss(line);
	ss >> word;
	
	// Routine metadata.
	if (word == "ROUTINE_NAME")
	{
		if (ss >> word)
			r_name = word;
		
		line = "";
	}
	else if (word == "ROUTINE_OBJECT_NAME")
	{
		if (ss >> word)
			r_object_name = word;
		
		line = "";
	}
	else if (word == "ROUTINE_INCLUDE")
	{
		if (ss >> word)
			r_includes.push_back(word);
		
		line = "";
	}
	
	// Routine parameters.
	else if (word == "ROUTINE_REQUIRE")
	{
		word2 = "";
		while (ss >> word)
			word2 += word + " ";
		if (word2.length() > 0)
			word2.pop_back();
		r_params.push_back(word2);
		
		line = "";
	}
	
	// Routine's number of blocks in CUDA kernel launch.
	else if (word == "ROUTINE_NBLOCKS")
	{
		word2 = "(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK";
		if (ss >> word2)
			r_nblocks = word2;
		
		line = "";
	}
	
	// Routine's thread-block size in CUDA kernel launch.
	else if (word == "ROUTINE_BLOCKSIZE")
	{
		word2 = "M_TBLOCK";
		if (ss >> word2)
			r_blocksize = word2;
		
		line = "";
	}
	
	// Kernel parameters.
	else if (word == "KERNEL_REQUIRE")
	{
		// Get the kernel parameter (only one, defined up till '|').
		word2 = "";
		while (ss >> word && word != "|")
		{
			word2 += word + " ";
			last_word = word;
		}
		if (word2.size() > 0)
		{
			word2.pop_back();
			r_kernel_params.push_back(word2);
			
			// Get the possible inputs for this parameter. 
			words3.clear();
			while (ss >> word)
				words3.push_back(word);
			if (words3.size() == 0)
				words3.push_back(last_word);
			r_kernel_inputs.push_back(words3);
		}
		
		line = "";
	}
	
	// Routine template parameters.
	else if (word == "ROUTINE_TEMPLATE_PARAMS")
	{
		// Should appear only once in the input file. Put all arguments in one line.
		r_template_params += ", ";
		while (ss >> word)
			r_template_params += word + " ";
		if (r_template_params.length() > 0)
			r_template_params.pop_back();
		
		line = "";
	}
	else if (word == "ROUTINE_TEMPLATE_VALS")
	{
		// There should be as many vals on each line as there are params.
		word2 = "ufloat_t,ufloat_g_t,AP,";
		while (ss >> word)
			word2 += word;
		if (word2.length() > 0)
			r_template_vals.push_back(word2);
		
		line = "";
	}
	else if (word == "ROUTINE_TEMPLATE_ARGS")
	{
		// There should be as many valvars as vals.
		word2 = "";
		while (ss >> word)
			word2 += word + " ";
		if (word2.length() > 0)
		{
			word2.pop_back();
			r_template_args.push_back(word2);
		}
		
		line = "";
	}
	else
		;
	
	return 0;
}

std::string MakeRoutineTemplateHeader(std::string &r_template_params)
{
	std::string s = "";
	if (r_template_params.size() > 0)
		s = "template <" + r_template_params + ">\n";
	
	return s;
}

std::string MakeSetName(std::string &set_label)
{
	std::string s = "";
	if (set_label.length() > 0)
		s = std::string("_") + set_label;
	
	return s;
}

std::string MakeKernelInputs(int k, std::vector<std::vector<std::string>> &r_kernel_inputs)
{
	std::vector<std::string> inputs_k;
	for (int i = 0; i < r_kernel_inputs.size(); i++)
	{
		if (k < r_kernel_inputs[i].size())
			inputs_k.push_back( r_kernel_inputs[i][k] );
		else
			inputs_k.push_back( r_kernel_inputs[i].back() ); 
	}
	
	return Commatize(inputs_k, ", ");
}

std::string MakeIncludes(std::vector<std::string> &r_includes)
{
	std::string s = "";
	for (int k = 0; k < r_includes.size(); k++)
		s += "#include " + r_includes[k] + "\n";
	
	return s;
}

int GenerateRoutine(ImportedSet &iset, std::string set_label, std::string &s_in,
	std::string &r_name,
	std::string &r_object_name,
	std::vector<std::string> &r_params,
	std::vector<std::string> &r_includes,
	std::vector<std::string> &r_kernel_params,
	std::vector<std::vector<std::string>> &r_kernel_inputs,
	std::string &r_template_params,
	std::vector<std::string> &r_template_vals,
	std::vector<std::string> &r_template_args,
	std::string &r_nblocks,
	std::string &r_blocksize
)
{
	std::string s_out = "";
	
	// If there were not template values for this code, generate one default instance.
	if (r_template_vals.size() == 0)
	{
		//r_template_params = "const ArgsPack *AP";
		r_template_args.push_back("mesh->n_ids[i_dev][L] > 0");
		r_template_vals.push_back("ufloat_t,ufloat_g_t,AP");
	}
	
	
	
	// CUDA kernel.
	s_out += MakeIncludes(r_includes) + "\n";
	s_out += MakeRoutineTemplateHeader(r_template_params);
	s_out += "__global__\n";
	s_out += "void Cu_" + r_name + MakeSetName(set_label) + "(";
	//s_out += "double A";
	s_out += Commatize(r_kernel_params, ",");
	s_out += ")\n";
	s_out += "{\n";
	s_out += s_in;
	s_out += "}\n\n";
	
	// C++ routine calling the kernel.
	s_out += "template <typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP, const LBMPack *LP>\n";
	s_out += "int " + r_object_name + "<ufloat_t,ufloat_g_t,AP,LP>::S_" + r_name + MakeSetName(set_label) + "(";
	s_out += Commatize(r_params,", ");
	s_out += ")\n";
	s_out += "{\n";
	for (int k = 0; k < r_template_vals.size(); k++)
	{
		s_out += "\tif (" + r_template_args[k] + ")\n";
		s_out += "\t{\n";
		s_out += "\t\tCu_" + r_name + MakeSetName(set_label) + "<" + r_template_vals[k] + "><<<" + r_nblocks + "," + r_blocksize + ",0,mesh->streams[i_dev]>>>(" + MakeKernelInputs(k,r_kernel_inputs) + ");\n";
		s_out += "\t}\n";
	}
	s_out += "\n";
	s_out += "\treturn 0;\n";
	s_out += "}\n";
	
	// Update final string.
	s_in = s_out;
	
	return 0;
}

int Process_OutputGenerateRoutine(ImportedSet &iset, std::string set_label, std::string &s_in)
{
	// Parameters.
	int indent = 0;
	std::string line = "";
	std::vector<int> open_fors;
	std::stringstream ss = std::stringstream(s_in);
	std::string r_name;
	std::string r_object_name;
	std::vector<std::string> r_params;
	std::vector<std::string> r_includes;
	std::vector<std::string> r_kernel_params;
	std::vector<std::vector<std::string>> r_kernel_inputs;
	std::string r_template_params = "typename ufloat_t, typename ufloat_g_t, const ArgsPack *AP";
	std::vector<std::string> r_template_vals;
	std::vector<std::string> r_template_args;
	std::string r_nblocks = "(M_LBLOCK+mesh->n_ids[i_dev][L]-1)/M_LBLOCK";
	std::string r_blocksize = "M_TBLOCK";
	
	s_in = "";
	while (ss.peek() != EOF)
	{
		line = CollectLineUnformatted(ss);
		ProcessLineGetRoutineParameters(line,
			r_name,
			r_object_name,
			r_params,
			r_includes,
			r_kernel_params,
			r_kernel_inputs,
			r_template_params,
			r_template_vals,
			r_template_args,
			r_nblocks,
			r_blocksize
		);
		s_in += line;
	}
	
	// Generate the routine.
	GenerateRoutine(iset, set_label, s_in,
		r_name,
		r_object_name,
		r_params,
		r_includes,
		r_kernel_params,
		r_kernel_inputs,
		r_template_params,
		r_template_vals,
		r_template_args,
		r_nblocks,
		r_blocksize
	);
	
	return 0;
}
