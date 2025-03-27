#include "gen.h"

int AcquireTextSubs(std::map<std::string,std::string> &map_sub, std::string &line)
{
	// Parameters.
	std::string word = "";
	std::string word2 = "";
	std::stringstream ss(line);
	ss >> word;
	
	if (word == "TEXTSUBS")
	{
		ss >> word; // word now stores the key.
		ReplaceSubstring(line,"TEXTSUBS ","");
		ReplaceSubstring(line,word + " ","");
		ReplaceSubstring(line,"\n","");
		if (ss >> word2)
			map_sub[word] = line;
		line = "";
	}
	
	return 0;
}

int Process_TextSubs(std::map<std::string,std::string> &map_sub, std::string &line)
{
	for (const auto &pair : map_sub)
		ReplaceSubstring(line,pair.first,pair.second);
	
	return 0;
}

int Process_TextSubsFlag(std::map<std::string,std::string> &map_sub, std::string &line)
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
				if (flags[k] == 't')
					Process_TextSubs(map_sub,line);
			}
		}
	}
	
	return 0;
}

int Process_OutputTextSubs(std::string &s_in)
{
	// Parameters.
	std::string line = "";
	std::stringstream ss = std::stringstream(s_in);
	std::map<std::string,std::string> map_sub;
	
	s_in = "";
	while (ss.peek() != EOF)
	{
		line = CollectLineFormatted(ss);
		AcquireTextSubs(map_sub, line);
		Process_TextSubsFlag(map_sub, line);
		s_in += line;
	}
	
	return 0;
}
