#ifndef GEN_H
#define GEN_H

#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>
#include <sstream>
#include <map>
#include <algorithm>
#include "exprtk.hpp"
#include <chrono>
#include <ctime>

// o====================================================================================
// | Import class.
// o====================================================================================

// Format of imported data:
// vector (N0+1) VECTOR_NAME_0
// (v0)_0 (v1)_0 (v2)_0 ... (vN0)_0
// vector N1 VECTOR_NAME_1
// (v0)_1 (v1)_1 (v2)_1 ... (vN1)_1
// .
// .
// .
// matrix (N0+1) (M0+1) MATRIX_NAME_0
// (m_00)_0 (m_01)_0 (m_02)_0 ... (m_0N0)_0
// (m_10)_0 (m_11)_0 (m_12)_0 ... (m_1N0)_0
// .
// .
// (m_M00)_0 (m_M01)_0 (m_M02)_0 ... (m_M0N0)_0
// .
// .
// .

class ImportedSet
{
	public:
	
	std::vector<std::string> sets;
	std::vector<std::string> names;
	std::map<std::string, std::string> imported_types;
	std::map<std::string, std::string> imported_values;
	std::map<std::string, std::string> imported_setvalues;
	std::map<std::string, std::vector<std::string>> imported_vectors;
	std::map<std::string, std::vector<std::string>> imported_setvectors;
	
	ImportedSet()
	{
	}
	
	int ImportFromFile(std::string filename)
	{
		// Parameters.
		std::string word = "";
		std::string name = "";
		std::string mapped = "";
		int n;
		double d;
		
		// Open the file, and start adding the data.
		std::ifstream in = std::ifstream("./import/" + filename + ".txt");
		while (in.peek() != EOF)
		{
			// Read a word. This determines the course of action.
			in >> word;
			
			// Add a number of sets.
			// Each set produces a separate file and determines the proper labels to apply to data.
			if (word == "sets")
			{
				in >> n;
				for (int k = 0; k < n; k++)
				{
					in >> mapped;
					sets.push_back(mapped);
					//std::cout << "Added a set: " << mapped << std::endl;
				}
			}
			
			// Add a data label. This is appended by a set name, if applicable.
			if (word == "name")
			{
				in >> name >> mapped;
				names.push_back(name);
				imported_types[name] = mapped;
				//std::cout << "Added a name: " << name << std::endl;
			}
			
			// Add a value by name.
			if (word == "value")
			{
				in >> name >> mapped;
				imported_values[name] = mapped;
				//std::cout << "Added a value: " << name << ":" << mapped << std::endl;
			}
			
			// Add a value by name.
			if (word == "setvalue")
			{
				in >> name >> mapped;
				imported_setvalues[name] = mapped;
				//std::cout << "Added a value: " << name << ":" << mapped << std::endl;
			}
			
			// Add a set of data. This is in the form of a map from a data label to a vector.
			if (word == "vector")
			{
				in >> name >> n;
				std::vector<std::string> data;
				
				for (int k = 0; k < n; k++)
				{
					in >> mapped;
					data.push_back(mapped);
				}
				
				imported_vectors[name] = data;
				//std::cout << "Added a vector (size " << n << "): " << name <<  std::endl;
			}
			
			// Add a set of data. This is in the form of a map from a data label to a vector.
			if (word == "setvector")
			{
				in >> name >> n;
				std::vector<std::string> data;
				
				for (int k = 0; k < n; k++)
				{
					in >> mapped;
					data.push_back(mapped);
				}
				
				imported_setvectors[name] = data;
				//std::cout << "Added a setvector (size " << n << "): " << name <<  std::endl;
			}
			
			// Reset strings.
			word = "";
			name = "";
			mapped = "";
		}
		
		return 0;
	}
};



// o====================================================================================
// | Global variables.
// o====================================================================================

// Make Exptrk expression and parse once and re-use.
extern exprtk::expression<double> expression;
extern exprtk::parser<double> parser;
extern char c_buff[100];



// o====================================================================================
// | Overloaded operators.
// o====================================================================================

// Credit: NutCracker (https://stackoverflow.com/a/55124613).
template <typename T> 
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) 
{ 
    os << "[";
    for (int i = 0; i < v.size(); ++i) { 
        os << v[i]; 
        if (i != v.size() - 1) 
            os << ", "; 
    }
    os << "]\n";
    return os; 
}



// o====================================================================================
// | Routines.
// o====================================================================================

// Text Substitutions.
int               AcquireTextSubs(std::map<std::string,std::string> &map_sub, std::string &line);
int               Process_TextSubs(std::map<std::string,std::string> &map_sub, std::string &line);
int               Process_TextSubsFlag(std::map<std::string,std::string> &map_sub, std::string &line);
int               Process_OutputTextSubs(std::string &s_in);

// String.
int               CountSubstring(std::string &s, std::string subs);
int               ReplaceSubstring(std::string &s, std::string s_old, std::string s_new);
int               ReplaceFirstSubstring(std::string &s, std::string s_old, std::string s_new, size_t pos);
std::string       Commatize(std::vector<std::string> s_vec, std::string comma);
int               ReverseCommatize(std::string &line, std::vector<std::string> &vec, std::string op);
std::string       ReadOneLine(std::istream &in);
std::string       CollectLine(std::istream &in);
std::string       CollectLineUnformatted(std::istream &in);
std::string       CollectLineFormatted(std::istream &in);
std::string       GetContentBetweenLimitChars(std::string s, char c1, char c2);
std::string       GetContentBetweenLimitStrings(std::string s, std::string s1, std::string s2);
int               ReplaceStringsBetweenLimitStrings(std::string &s, std::string e, std::string r, std::string s1, std::string s2);

// Loop.
int               ProcessLineIfAndFor(ImportedSet &iset, std::string set_label, std::istream &in, std::string &line);
int               ProcessLoopStatements(ImportedSet &iset, std::string set_label, std::string &line);
int               ProcessFor(ImportedSet &iset, std::string set_label, std::istream &in, std::string &line);
int               ProcessIf(ImportedSet &iset, std::string set_label, std::istream &in, std::string &line);
int               Process_OutputInLoops(ImportedSet &iset, std::string set_label, std::string &s_in);

// Line.
int               ProcessLineRegular(ImportedSet &iset, std::string set_label, std::istream &in, std::string &line);
int               ApplyAllOperations(ImportedSet &iset, std::string set_label, std::string &line);
int               ProcessLineSetLabels(ImportedSet &iset, std::string set_label, std::string &line, int var=1);
int               ProcessLineMath(std::string &line);
int               ProcessLineSumsAndProds(std::string &line);
int               ProcessLineGetRidOfZerosSum(std::string &line);
int               ProcessLineConditionConstructor(std::string &line);
int               ProcessLineSimplifyAndsOrs( std::string &line);
int               Process_OutputLineSubs(ImportedSet &iset, std::string set_label, std::string &s_in);

// Other Line Substitutions.
int               ProcessLineGridIndex(std::string &line);

// Templates.
int               Template_CollectData(std::istream &in, std::string &line);
int               Process_OutputTemplates(std::string &s_in);
std::string       Template_PrimaryMode(std::string s_in, std::vector<std::string> args);
std::string       Template_ExtrapolateHalo(std::vector<std::string> args, std::vector<std::string> args2);

// Output.
std::string       make_indent(int indents, std::string tab);
int               ProcessLineFormatInterpret(ImportedSet &iset, std::string set_label, std::map<std::string,std::string> &fparams, std::string &line, int &indent, std::vector<int> &open_fors);
int               Process_OutputFormatting(ImportedSet &iset, std::string set_label, std::map<std::string,std::string> &fparams, std::string &in);

// Routine.
int               GenerateRoutine(ImportedSet &iset, std::string set_label, std::string &s_in);
int               Process_OutputGenerateRoutine(ImportedSet &iset, std::string set_label, std::string &s_in);

// Main.
std::string       Banner();

#endif
