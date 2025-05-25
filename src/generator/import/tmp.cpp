#include <iostream>
#include <fstream>

int main(int argc, char *argv[])
{
	std::ifstream in1 = std::ifstream("mat_interp_2D_single.txt");
	std::ifstream in2 = std::ifstream("mat_interp_3D_single.txt");
	std::ofstream out = std::ofstream("InterpMatrices.txt");

	out << "vector InterpM2D 256 ";
	std::string s;
	for (int i = 0; i < 16; i++)
	{
		for (int j = 0; j < 16; j++)
		{
			in1 >> s;
			out << s << " ";
		}
		out << std::endl;
	}
	out << std::endl;

	out << "vector InterpM3D 4096 ";
	for (int i = 0; i < 64; i++)
	{
		for (int j = 0; j < 64; j++)
		{
			in2 >> s;
			out << s << " ";
		}
		out << std::endl;
	}

	return 0;
}
