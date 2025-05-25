#include "gen.h"

int CountSubstring(std::string &s, std::string subs)
{
	int N = 0;
	size_t n = s.find(subs);
	while (n != std::string::npos)
	{
		N++;
		n = s.find(subs,n+1);
	}
	
	return N;
}

// Replace all instances of 's_old' in the string 's' with 's_new'.
int ReplaceSubstring(std::string &s, std::string s_old, std::string s_new)
{
	size_t n = s.find(s_old);
	int l_old = s_old.size();
	while (n != std::string::npos)
	{
		s = s.substr(0,n) + s_new + s.substr(n+l_old,s.size());
		n = s.find(s_old);
	}
	
	return 0;
}

// Replace first instance of 's_old' in the string 's' with 's_new'.
int ReplaceFirstSubstring(std::string &s, std::string s_old, std::string s_new, size_t pos)
{
	size_t n = s.find(s_old,pos);
	int l_old = s_old.size();
	if (n != std::string::npos)
		s = s.substr(0,n) + s_new + s.substr(n+l_old,s.size());
	
	return 0;
}

std::string Commatize(std::vector<std::string> s_vec, std::string comma)
{
	// If empty vector, return an empty string.
	if (s_vec.size() == 0)
		return "";
	
	// If only one element, just return that element.
	if (s_vec.size() == 1)
		return s_vec[0];
	
	// If more than one element, then commatize the list of strings.
	// i.e. add a 'comma' between them to make one string, where 'comma' can be any other string.
	std::string s = s_vec[0];
	for (int k = 1; k < s_vec.size(); k++)
		s = s + comma + s_vec[k];
	
	return s;
}

int ReverseCommatize(std::string &line, std::vector<std::string> &vec, std::string op)
{
	ReplaceSubstring(line, op, " ");
	std::stringstream ss(line);
	std::string word;
	while (ss >> word)
		vec.push_back(word);
	
	return 0;
}

std::string ReadOneLine(std::istream &in)
{
	// Parameters.
	std::string line = "";
	std::string w = "";
	std::vector<std::string> w_vec;
	
	// Read the line and remove extra white space.
	std::getline(in, line);
	std::stringstream ss(line);
	line = "";
	while (ss >> w)
		w_vec.push_back(w);
	line = Commatize(w_vec, " ");
	
	return line;
}

std::string CollectLine(std::istream &in)
{
	std::string word = "";
	std::stringstream ss;
	std::string line = ReadOneLine(in);
	
	// If empty line, skip.
	if (line == "")
		return "";
	
	// If comment, skip.
	ss = std::stringstream(line);
	if (ss >> word && word[0]=='#')
		return "";
	
	return line + "\n";
}

std::string CollectLineUnformatted(std::istream &in)
{
	std::string line = "";
	std::getline(in,line);
	
	return line + "\n";
}

std::string CollectLineFormatted(std::istream &in)
{
	std::string line = ReadOneLine(in);
	
	// If empty line, skip.
	if (line == "")
		return "";
	
	// If last character was a backslash, add the next line to this one.
	if (line[line.size()-1] == '\\')
	{
		line = line.substr(0,line.size()-1) + " " + CollectLineFormatted(in);
		return line;
	}
	else
		return line + "\n";
}

std::string GetContentBetweenLimitChars(std::string s, char c1, char c2)
{
	// Parameters.
	std::stringstream ss(s);
	std::string content = "";
	char c = '\0';
	int c1_counter = 1;
	int c2_counter = 0;
	
	while (!(c == c2 && c1_counter==c2_counter))
	{
		if (c != '\0')
			content = content + c;
		c = ss.get();
		
		if (c == c1)
			c1_counter++;
		if (c == c2)
			c2_counter++;
	}
	
	return content;
}

std::string GetContentBetweenLimitStrings(std::string s, std::string s1, std::string s2)
{
	// Parameters.
	size_t n = s.find(s2);
	int s1_counter = 1;
	int s2_counter = 0;
	std::string s_before = "";
	std::string content = "";
	
	while (n != std::string::npos && s1_counter != s2_counter)
	{
		s_before = s.substr(0,n);
		s1_counter = CountSubstring(s_before,s1);
		s2_counter = CountSubstring(s_before,s2);
		std::cout << "Checking |" << s_before << "|, found " << s1 << "=" << s1_counter << ", " << s2 << "=" << s2_counter << std::endl;
		if (s1_counter-s2_counter == 0)
			content = s_before;
		else
			n = s.find(s2,n+1);
	}
	
	return content;
}

// Replaces the characters in a string written as: 's1s2s3s4s5...snc2'.
std::string ReplaceCharsBetweenLimitChars(std::string s, char e, char r, char c1, char c2)
{
	// Parameters.
	std::stringstream ss(s);
	std::string content = "";
	char c = '\0';
	int c1_counter = 1;
	
	while (!(c == c2 && c1_counter == 0))
	{
		if (c1_counter==1 && c == e)
			c = r;
		if (c != '\0')
			content = content + c;
		c = ss.get();
		
		if (c == c1)
			c1_counter++;
		if (c == c2)
			c1_counter--;
	}
	
	return content;
}

// Replaces the substrings in a string written as: 's1s2s3s4s5...snc2'.
int ReplaceStringsBetweenLimitStrings(std::string &s, std::string e, std::string r, std::string s1, std::string s2)
{
	// Parameters.
	size_t n = s.find(e);
	int s1_counter = 0;
	int s2_counter = 0;
	std::string s_before = "";
	std::string content = s;
	
	// Perform the replacements.
	while (n != std::string::npos)
	{
		s_before = content.substr(0,n);
		s1_counter = CountSubstring(s_before,s1);
		s2_counter = CountSubstring(s_before,s2);
		if (s1_counter-s2_counter == 0)
		{
			ReplaceFirstSubstring(content,e,r,n);
			n = content.find(e,n);
		}
		else
			n = content.find(e,n+1);
	}
	s = content;
	
	return 0;
}

std::string GetContentBetweenLimitStrings(std::istream &in, std::string s1, std::string s2)
{
	// Parameters.
	std::string word = "";
	std::string content = "";
	std::string line_p = "";
	std::stringstream ss;
	int s1_counter = 1;
	int s2_counter = 0;
	
	while (!(word == s2 && s1_counter==s2_counter))
	{
		content = content + line_p;
		ss.clear();
		line_p = CollectLine(in);
		ss = std::stringstream(line_p);
		ss >> word;
		
		size_t n_s1 = line_p.find(s1);
		if (n_s1 != std::string::npos && line_p[n_s1-1] != '_')
			s1_counter++;
		if (line_p.find(s2) != std::string::npos)
			s2_counter++;
	}
	
	return content;
}
