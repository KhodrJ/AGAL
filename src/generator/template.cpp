#include "gen.h"

std::string Template_PrimaryMode(std::vector<std::string> args, std::vector<std::string> args2)
{
	// args i: 0 - the block of text in the primary access loop
	// args2 are the conditions for processing each block
	
// 	s_ID_cblock[threadIdx.x] = -1;
// 	if ((threadIdx.x < M_LBLOCK)and(kap < n_ids_idev_L))
// 	{
// 		s_ID_cblock[threadIdx.x] = id_set_idev_L[kap];
// 	}
// 	__syncthreads();
// 
// 	// Loop over block Ids.
// 	for (int k = 0; k < M_LBLOCK; k += 1)
// 	{
// 		i_kap_b = s_ID_cblock[k];
// 
// 		// This part is included if n>0 only.
// 		if (i_kap_b > -1)
// 		{
// 			i_kap_bc=cblock_ID_nbr_child[i_kap_b];
// 			block_on_boundary=cblock_ID_mask[i_kap_b];
// 		}
// 
// 		// Latter condition is added only if n>0.
// 		if (i_kap_b > -1 && ((i_kap_bc<0)||(block_on_boundary==1)))
// 		{
// 			
// 		}
// 	}
	std::string cond = "";
	if (args2.size() > 0)
		cond = args2[0];
	if (cond != "")
		cond = "&&(" + cond + ")";
	
	std::string t = "\n";
	t += "REG s_ID_cblock[threadIdx.x] = -1;\n";
	t += "OUTIF ((threadIdx.x<M_LBLOCK)and(kap<n_ids_idev_L))\n";
	t += "REG s_ID_cblock[threadIdx.x] = id_set_idev_L[kap];\n";
	t += "END_OUTIF\n";
	t += "REG __syncthreads();\n";
	t += "<\n";
	t += "// Loop over block Ids.\n";
	t += "OUTFOR k 1   0 M_LBLOCK 1\n";
	t += "REG i_kap_b = s_ID_cblock[k];\n";
	t += "<\n";
	
	if (args2.size() > 1)
	{
		t += "// Load data for conditions on cell-blocks.\n";
		t += "OUTIF (i_kap_b>-1)\n";
		for (int k = 1; k < args2.size(); k++)
			t += args2[k] + "\n";
		//if (cond.find("i_kap_bc") != std::string::npos)
		//	t += "REG i_kap_bc=cblock_ID_nbr_child[i_kap_b];\n";
		//if (cond.find("block_on_boundary") != std::string::npos)
		//	t += "REG block_on_boundary=cblock_ID_mask[i_kap_b];\n";
		t += "END_OUTIF\n";
		t += "<\n";
	}
	
	t += "// Latter condition is added only if n>0.\n";
	t += "OUTIF ((i_kap_b>-1)"+cond+")\n";
	t += args[0];
	t += "END_OUTIF\n";
	t += "END_OUTFOR\n";
	
	return t;
}

std::string Template_PrimaryModeUpgraded(std::vector<std::string> args, std::vector<std::string> args2)
{
	std::string cond = "";
	if (args2.size() > 0)
		cond = args2[0];
	if (cond != "")
		cond = "&&(" + cond + ")";
	
	std::string t = "\n";
	t += "OUTIF (threadIdx.x<M_LWBLOCK)\n";
	t += "REG s_ID_cblock[threadIdx.x] = -1;\n";
	t += "OUTIFL (kap<n_ids_idev_L)\n";
	t += "REG s_ID_cblock[threadIdx.x] = id_set_idev_L[kap];\n";
	t += "END_OUTIFL\n";
	t += "END_OUTIF\n";
	t += "REG __syncthreads();\n";
	t += "REG i_kap_b = s_ID_cblock[threadIdx.x/AP->M_WBLOCK];\n";
	t += "<\n";
	
	if (args2.size() > 1)
	{
		t += "// Load data for conditions on cell-blocks.\n";
		t += "OUTIF (i_kap_b>-1)\n";
		for (int k = 1; k < args2.size(); k++)
			t += args2[k] + "\n";
		t += "END_OUTIF\n";
		t += "<\n";
	}
	
	t += "// Latter condition is added only if n>0.\n";
	t += "OUTIF ((i_kap_b>-1)"+cond+")\n";
	t += args[0];
	t += "END_OUTIF\n";
	
	return t;
}

std::string Template_ExtrapolateHalo(std::vector<std::string> args, std::vector<std::string> args2)
{
	std::string t = "";
	
	return t;
}

int Template_CollectData(std::istream &in, std::string &line)
{
	std::string word = "";
	std::string word2 = "";
	std::string line_p;
	std::string collected = "";
	std::string name = "";
	std::vector<std::string> args;
	std::vector<std::string> args2;
	std::stringstream ss(line);
	ss >> word;
	
	// Process a template: collect lines of data until END_TEMPLATE, then format accordingly.
	if (word == "TEMPLATE")
	{
		// Collect lines of data.
		while (word != "END_TEMPLATE")
		{
			line_p = CollectLine(in);
			ss.clear();
			ss = std::stringstream(line_p);
			ss >> word;
			
			if (word != "END_TEMPLATE")
			{
				ss >> word2;
				
				// Process additional template parameters.
				if (word == "TEMPLATE")
				{
					if (word2 == "NAME")
					{
						ss >> word2;
						name = word2;
						//std::cout << "Detected template name: " << name << std::endl;
					}
					if (word2 == "ARG")
					{
						word2 = "";
						while (ss >> word)
							word2 += word + " ";
						if (word2.length() > 0)
							word2.pop_back();
						args2.push_back(word2);
						//std::cout << "Detected new template arg: " << word2 << std::endl;
					}
					if (word2 == "NEW_BLOCK")
					{
						args.push_back(collected);
						//std::cout << "Detected end of current block:\n" << collected << std::endl;
						collected = "";
					}
				}
				else // Otherwise, just collect the current line.
					collected += line_p;
			}
		}
		
		// Now choose the template based on the stored name (one must be provided).
		if (name == "PRIMARY_ORIGINAL")
			line = Template_PrimaryMode(args, args2);
		else if (name == "PRIMARY_UPGRADED")
			line = Template_PrimaryModeUpgraded(args, args2);
		else // Default: just return the block(s) of text.
			line = Commatize(args,"\n");
	}
	
	return 0;
}

int Process_OutputTemplates(std::string &s_in)
{
	// Parameters.
	std::string line = "";
	std::stringstream ss = std::stringstream(s_in);
	
	s_in = "";
	while (ss.peek() != EOF)
	{
		line = CollectLine(ss);
		Template_CollectData(ss,line);
		s_in += line;
	}
	
	return 0;
}
