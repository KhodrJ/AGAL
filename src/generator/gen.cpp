#include "gen.h"

exprtk::expression<double> expression;
exprtk::parser<double> parser;
char c_buff[100];

std::string Banner()
{
	// Get the current date and time.
	// Credit: G S (https://stackoverflow.com/a/27856440).
	auto t = std::chrono::system_clock::now();
	std::time_t time = std::chrono::system_clock::to_time_t(t);
	std::string s_time = std::ctime(&time);
	int n_c = s_time.length()+16; s_time[s_time.length()-1] = ' ';
	for (int i = n_c; i < 85-1; i++)
		s_time = s_time + " ";
	
	// Make the string.
	std::string s = "";
	s = s + "/**************************************************************************************/\n";
	s = s + "/*                                                                                    */\n";
	s = s + "/*  Author: Khodr Jaber                                                               */\n";
	s = s + "/*  Affiliation: Turbulence Research Lab, University of Toronto                       */\n";
	s = s + "/*  Last Updated: " + s_time + "*/\n";
	s = s + "/*                                                                                    */\n";
	s = s + "/**************************************************************************************/\n";
	return s;
}

int main(int argc, char *argv[])
{
	// Parameters.
	std::stringstream ss;
	std::string filename = "";
	std::string s_in = "";
	std::string line = "";
	std::map<std::string,std::string> fparams = std::map<std::string,std::string>{
		{"name", "out"},
		{"dir","./output/"}
	};
	
	// Open files.
	filename = "./input/test.agc";
	if (argc > 1)
		filename = std::string(argv[1]);
	std::ifstream in = std::ifstream(filename);
	if (!in)
	{
		std::cout << "ERROR: Could not open the requested input file." << std::endl;
		return -1;
	}
	std::cout << "Generating the scripts for " << filename << "..." << std::endl;
	std::ofstream out;
	//std::ofstream out = std::ofstream("./output/out.txt");
	ImportedSet iset;
	iset.ImportFromFile("LBM");
	iset.ImportFromFile("InterpMatrices");
	
	// For each set in the ImportedSet, produce an output file. If there are no sets, then just one default output file.
	if (iset.sets.size() == 0)
		iset.sets.push_back("");
	for (int k = 0; k < iset.sets.size(); k++)
	{
		// Retrieve the input from the file.
		std::cout << "Doing set (" << iset.sets[k] << ")..." << std::endl;
		while (in.peek() != EOF)
		{
			line = CollectLine(in);
			s_in += line;
		}
		
		// Process text substitutions.
		Process_OutputTextSubs(s_in);
		
		// Process the in-code for- and if- loops.
		Process_OutputInLoops(iset, iset.sets[k], s_in);
		
		// Apply templates.
		Process_OutputTemplates(s_in);
		
		// Process set labels.
		Process_OutputLineSubs(iset, iset.sets[k], s_in);
		
		// Process formatting.
		Process_OutputFormatting(iset, iset.sets[k], fparams, s_in);
		
		// Process routine structure.
		Process_OutputGenerateRoutine(iset, iset.sets[k], s_in);
		
		// Open the output file for this set, then write to it.
		filename = fparams["dir"] + fparams["name"] + "_" + iset.sets[k] + ".cu";
		out.open(filename);
		out << Banner() << std::endl;
		out << s_in << std::endl;
		
		// Close the output file and reset parameters for next set.
		out.close();
		in.clear();
		in.seekg(0);
		s_in = "";
	}
	
	return 0;
}
