// Remove LBM_nexchange, LBM_pexc

std::vector<std::map<std::string, std::string>> LBM_Variables_Preprocess
{
	{
		// D2Q9
		{"LBM_name", "d2q9"},
		{"LBM_dim", "2"},
		{"LBM_size", "9"},
		{"LBM_conn_size", "9"},
		{"LBM_3D", "0"}
	},
	{
		// D3Q19
		{"LBM_name", "d3q19"},
		{"LBM_dim", "3"},
		{"LBM_size", "19"},
		{"LBM_conn_size", "27"},
		{"LBM_3D", "1"}
	},
	{
		// D3Q27
		{"LBM_name", "d3q27"},
		{"LBM_dim", "3"},
		{"LBM_size", "27"},
		{"LBM_conn_size", "27"},
		{"LBM_3D", "1"}
	}
};

std::vector<std::map<std::string, std::string>> LBM_Variables_Postprocess
{
	{
		// D2Q9
		{"LBM_w(0)", "^D< 4.0/9.0 >^"},
		{"LBM_w(1)", "^D< 1.0/9.0 >^"},
		{"LBM_w(2)", "^D< 1.0/9.0 >^"},
		{"LBM_w(3)", "^D< 1.0/9.0 >^"},
		{"LBM_w(4)", "^D< 1.0/9.0 >^"},
		{"LBM_w(5)", "^D< 1.0/36.0 >^"},
		{"LBM_w(6)", "^D< 1.0/36.0 >^"},
		{"LBM_w(7)", "^D< 1.0/36.0 >^"},
		{"LBM_w(8)", "^D< 1.0/36.0 >^"},
		{"LBM_pb(0)", "0"},
		{"LBM_pb(1)", "3"},
		{"LBM_pb(2)", "4"},
		{"LBM_pb(3)", "1"},
		{"LBM_pb(4)", "2"},
		{"LBM_pb(5)", "7"},
		{"LBM_pb(6)", "8"},
		{"LBM_pb(7)", "5"},
		{"LBM_pb(8)", "6"},
		{"LBM_c0(0)", "0.0"}, {"LBM_c1(0)", "0.0"}, {"LBM_c2(0)", "0.0"},
		{"LBM_c0(1)", "1.0"}, {"LBM_c1(1)", "0.0"}, {"LBM_c2(1)", "0.0"},
		{"LBM_c0(2)", "0.0"}, {"LBM_c1(2)", "1.0"}, {"LBM_c2(2)", "0.0"},
		{"LBM_c0(3)", "-1.0"}, {"LBM_c1(3)", "0.0"}, {"LBM_c2(3)", "0.0"},
		{"LBM_c0(4)", "0.0"}, {"LBM_c1(4)", "-1.0"}, {"LBM_c2(4)", "0.0"},
		{"LBM_c0(5)", "1.0"}, {"LBM_c1(5)", "1.0"}, {"LBM_c2(5)", "0.0"},
		{"LBM_c0(6)", "-1.0"}, {"LBM_c1(6)", "1.0"}, {"LBM_c2(6)", "0.0"},
		{"LBM_c0(7)", "-1.0"}, {"LBM_c1(7)", "-1.0"}, {"LBM_c2(7)", "0.0"},
		{"LBM_c0(8)", "1.0"}, {"LBM_c1(8)", "-1.0"}, {"LBM_c2(8)", "0.0"},
		{"LBM_nbr(0,0,0)", "0"},
		{"LBM_nbr(1,0,0)", "1"},
		{"LBM_nbr(0,1,0)", "2"},
		{"LBM_nbr(-1,0,0)", "3"},
		{"LBM_nbr(0,-1,0)", "4"},
		{"LBM_nbr(1,1,0)", "5"},
		{"LBM_nbr(-1,1,0)", "6"},
		{"LBM_nbr(-1,-1,0)", "7"},
		{"LBM_nbr(1,-1,0)", "8"},
		{"LBM_pexc(0)", "1"},
		{"LBM_pexc(1)", "2"},
		{"LBM_pexc(2)", "5"}
	},
	{
		// D3Q19
		{"LBM_w(0)", "^D< 1.0/3.0 >^"},
		{"LBM_w(1)", "^D< 1.0/18.0 >^"},
		{"LBM_w(2)", "^D< 1.0/18.0 >^"},
		{"LBM_w(3)", "^D< 1.0/18.0 >^"},
		{"LBM_w(4)", "^D< 1.0/18.0 >^"},
		{"LBM_w(5)", "^D< 1.0/18.0 >^"},
		{"LBM_w(6)", "^D< 1.0/18.0 >^"},
		{"LBM_w(7)", "^D< 1.0/36.0 >^"},
		{"LBM_w(8)", "^D< 1.0/36.0 >^"},
		{"LBM_w(9)", "^D< 1.0/36.0 >^"},
		{"LBM_w(10)", "^D< 1.0/36.0 >^"},
		{"LBM_w(11)", "^D< 1.0/36.0 >^"},
		{"LBM_w(12)", "^D< 1.0/36.0 >^"},
		{"LBM_w(13)", "^D< 1.0/36.0 >^"},
		{"LBM_w(14)", "^D< 1.0/36.0 >^"},
		{"LBM_w(15)", "^D< 1.0/36.0 >^"},
		{"LBM_w(16)", "^D< 1.0/36.0 >^"},
		{"LBM_w(17)", "^D< 1.0/36.0 >^"},
		{"LBM_w(18)", "^D< 1.0/36.0 >^"},
		{"LBM_pb(0)", "0"},
		{"LBM_pb(1)", "2"},
		{"LBM_pb(2)", "1"},
		{"LBM_pb(3)", "4"},
		{"LBM_pb(4)", "3"},
		{"LBM_pb(5)", "6"},
		{"LBM_pb(6)", "5"},
		{"LBM_pb(7)", "8"},
		{"LBM_pb(8)", "7"},
		{"LBM_pb(9)", "10"},
		{"LBM_pb(10)", "9"},
		{"LBM_pb(11)", "12"},
		{"LBM_pb(12)", "11"},
		{"LBM_pb(13)", "14"},
		{"LBM_pb(14)", "13"},
		{"LBM_pb(15)", "16"},
		{"LBM_pb(16)", "15"},
		{"LBM_pb(17)", "18"},
		{"LBM_pb(18)", "17"},
		{"LBM_c0(0)", "0.0"}, {"LBM_c1(0)", "0.0"}, {"LBM_c2(0)", "0.0"},
		{"LBM_c0(1)", "1.0"}, {"LBM_c1(1)", "0.0"}, {"LBM_c2(1)", "0.0"},
		{"LBM_c0(2)", "-1.0"}, {"LBM_c1(2)", "0.0"}, {"LBM_c2(2)", "0.0"},
		{"LBM_c0(3)", "0.0"}, {"LBM_c1(3)", "1.0"}, {"LBM_c2(3)", "0.0"},
		{"LBM_c0(4)", "0.0"}, {"LBM_c1(4)", "-1.0"}, {"LBM_c2(4)", "0.0"},
		{"LBM_c0(5)", "0.0"}, {"LBM_c1(5)", "0.0"}, {"LBM_c2(5)", "1.0"},
		{"LBM_c0(6)", "0.0"}, {"LBM_c1(6)", "0.0"}, {"LBM_c2(6)", "-1.0"},
		{"LBM_c0(7)", "1.0"}, {"LBM_c1(7)", "1.0"}, {"LBM_c2(7)", "0.0"},
		{"LBM_c0(8)", "-1.0"}, {"LBM_c1(8)", "-1.0"}, {"LBM_c2(8)", "0.0"},
		{"LBM_c0(9)", "1.0"}, {"LBM_c1(9)", "0.0"}, {"LBM_c2(9)", "1.0"},
		{"LBM_c0(10)", "-1.0"}, {"LBM_c1(10)", "0.0"}, {"LBM_c2(10)", "-1.0"},
		{"LBM_c0(11)", "0.0"}, {"LBM_c1(11)", "1.0"}, {"LBM_c2(11)", "1.0"},
		{"LBM_c0(12)", "0.0"}, {"LBM_c1(12)", "-1.0"}, {"LBM_c2(12)", "-1.0"},
		{"LBM_c0(13)", "1.0"}, {"LBM_c1(13)", "-1.0"}, {"LBM_c2(13)", "0.0"},
		{"LBM_c0(14)", "-1.0"}, {"LBM_c1(14)", "1.0"}, {"LBM_c2(14)", "0.0"},
		{"LBM_c0(15)", "1.0"}, {"LBM_c1(15)", "0.0"}, {"LBM_c2(15)", "-1.0"},
		{"LBM_c0(16)", "-1.0"}, {"LBM_c1(16)", "0.0"}, {"LBM_c2(16)", "1.0"},
		{"LBM_c0(17)", "0.0"}, {"LBM_c1(17)", "1.0"}, {"LBM_c2(17)", "-1.0"},
		{"LBM_c0(18)", "0.0"}, {"LBM_c1(18)", "-1.0"}, {"LBM_c2(18)", "1.0"},
		{"LBM_c0(19)", "1.0"}, {"LBM_c1(19)", "1.0"}, {"LBM_c2(19)", "1.0"},
		{"LBM_c0(20)", "-1.0"}, {"LBM_c1(20)", "-1.0"}, {"LBM_c2(20)", "-1.0"},
		{"LBM_c0(21)", "1.0"}, {"LBM_c1(21)", "1.0"}, {"LBM_c2(21)", "-1.0"},
		{"LBM_c0(22)", "-1.0"}, {"LBM_c1(22)", "-1.0"}, {"LBM_c2(22)", "1.0"},
		{"LBM_c0(23)", "1.0"}, {"LBM_c1(23)", "-1.0"}, {"LBM_c2(23)", "1.0"},
		{"LBM_c0(24)", "-1.0"}, {"LBM_c1(24)", "1.0"}, {"LBM_c2(24)", "-1.0"},
		{"LBM_c0(25)", "-1.0"}, {"LBM_c1(25)", "1.0"}, {"LBM_c2(25)", "1.0"},
		{"LBM_c0(26)", "1.0"}, {"LBM_c1(26)", "-1.0"}, {"LBM_c2(26)", "-1.0"},
		{"LBM_nbr(0,0,0)", "0"},
		{"LBM_nbr(1,0,0)", "1"},
		{"LBM_nbr(-1,0,0)", "2"},
		{"LBM_nbr(0,1,0)", "3"},
		{"LBM_nbr(0,-1,0)", "4"},
		{"LBM_nbr(0,0,1)", "5"},
		{"LBM_nbr(0,0,-1)", "6"},
		{"LBM_nbr(1,1,0)", "7"},
		{"LBM_nbr(-1,-1,0)", "8"},
		{"LBM_nbr(1,0,1)", "9"},
		{"LBM_nbr(-1,0,-1)", "10"},
		{"LBM_nbr(0,1,1)", "11"},
		{"LBM_nbr(0,-1,-1)", "12"},
		{"LBM_nbr(1,-1,0)", "13"},
		{"LBM_nbr(-1,1,0)", "14"},
		{"LBM_nbr(1,0,-1)", "15"},
		{"LBM_nbr(-1,0,1)", "16"},
		{"LBM_nbr(0,1,-1)", "17"},
		{"LBM_nbr(0,-1,1)", "18"},
		{"LBM_nbr(1,1,1)", "19"},
		{"LBM_nbr(-1,-1,-1)", "20"},
		{"LBM_nbr(1,1,-1)", "21"},
		{"LBM_nbr(-1,-1,1)", "22"},
		{"LBM_nbr(1,-1,1)", "23"},
		{"LBM_nbr(-1,1,-1)", "24"},
		{"LBM_nbr(-1,1,1)", "25"},
		{"LBM_nbr(1,-1,-1)", "26"},
		{"LBM_pexc(0)", "1"},
		{"LBM_pexc(1)", "3"},
		{"LBM_pexc(2)", "5"},
		{"LBM_pexc(3)", "7"},
		{"LBM_pexc(4)", "9"},
		{"LBM_pexc(5)", "11"}
	},
	{
		// D3Q27
		{"LBM_w(0)", "^D< 8.0/27.0 >^"},
		{"LBM_w(1)", "^D< 2.0/27.0 >^"},
		{"LBM_w(2)", "^D< 2.0/27.0 >^"},
		{"LBM_w(3)", "^D< 2.0/27.0 >^"},
		{"LBM_w(4)", "^D< 2.0/27.0 >^"},
		{"LBM_w(5)", "^D< 2.0/27.0 >^"},
		{"LBM_w(6)", "^D< 2.0/27.0 >^"},
		{"LBM_w(7)", "^D< 1.0/54.0 >^"},
		{"LBM_w(8)", "^D< 1.0/54.0 >^"},
		{"LBM_w(9)", "^D< 1.0/54.0 >^"},
		{"LBM_w(10)", "^D< 1.0/54.0 >^"},
		{"LBM_w(11)", "^D< 1.0/54.0 >^"},
		{"LBM_w(12)", "^D< 1.0/54.0 >^"},
		{"LBM_w(13)", "^D< 1.0/54.0 >^"},
		{"LBM_w(14)", "^D< 1.0/54.0 >^"},
		{"LBM_w(15)", "^D< 1.0/54.0 >^"},
		{"LBM_w(16)", "^D< 1.0/54.0 >^"},
		{"LBM_w(17)", "^D< 1.0/54.0 >^"},
		{"LBM_w(18)", "^D< 1.0/54.0 >^"},
		{"LBM_w(19)", "^D< 1.0/216.0 >^"},
		{"LBM_w(20)", "^D< 1.0/216.0 >^"},
		{"LBM_w(21)", "^D< 1.0/216.0 >^"},
		{"LBM_w(22)", "^D< 1.0/216.0 >^"},
		{"LBM_w(23)", "^D< 1.0/216.0 >^"},
		{"LBM_w(24)", "^D< 1.0/216.0 >^"},
		{"LBM_w(25)", "^D< 1.0/216.0 >^"},
		{"LBM_w(26)", "^D< 1.0/216.0 >^"},
		{"LBM_pb(0)", "0"},
		{"LBM_pb(1)", "2"},
		{"LBM_pb(2)", "1"},
		{"LBM_pb(3)", "4"},
		{"LBM_pb(4)", "3"},
		{"LBM_pb(5)", "6"},
		{"LBM_pb(6)", "5"},
		{"LBM_pb(7)", "8"},
		{"LBM_pb(8)", "7"},
		{"LBM_pb(9)", "10"},
		{"LBM_pb(10)", "9"},
		{"LBM_pb(11)", "12"},
		{"LBM_pb(12)", "11"},
		{"LBM_pb(13)", "14"},
		{"LBM_pb(14)", "13"},
		{"LBM_pb(15)", "16"},
		{"LBM_pb(16)", "15"},
		{"LBM_pb(17)", "18"},
		{"LBM_pb(18)", "17"},
		{"LBM_pb(19)", "20"},
		{"LBM_pb(20)", "19"},
		{"LBM_pb(21)", "22"},
		{"LBM_pb(22)", "21"},
		{"LBM_pb(23)", "24"},
		{"LBM_pb(24)", "23"},
		{"LBM_pb(25)", "26"},
		{"LBM_pb(26)", "25"},
		{"LBM_c0(0)", "0.0"}, {"LBM_c1(0)", "0.0"}, {"LBM_c2(0)", "0.0"},
		{"LBM_c0(1)", "1.0"}, {"LBM_c1(1)", "0.0"}, {"LBM_c2(1)", "0.0"},
		{"LBM_c0(2)", "-1.0"}, {"LBM_c1(2)", "0.0"}, {"LBM_c2(2)", "0.0"},
		{"LBM_c0(3)", "0.0"}, {"LBM_c1(3)", "1.0"}, {"LBM_c2(3)", "0.0"},
		{"LBM_c0(4)", "0.0"}, {"LBM_c1(4)", "-1.0"}, {"LBM_c2(4)", "0.0"},
		{"LBM_c0(5)", "0.0"}, {"LBM_c1(5)", "0.0"}, {"LBM_c2(5)", "1.0"},
		{"LBM_c0(6)", "0.0"}, {"LBM_c1(6)", "0.0"}, {"LBM_c2(6)", "-1.0"},
		{"LBM_c0(7)", "1.0"}, {"LBM_c1(7)", "1.0"}, {"LBM_c2(7)", "0.0"},
		{"LBM_c0(8)", "-1.0"}, {"LBM_c1(8)", "-1.0"}, {"LBM_c2(8)", "0.0"},
		{"LBM_c0(9)", "1.0"}, {"LBM_c1(9)", "0.0"}, {"LBM_c2(9)", "1.0"},
		{"LBM_c0(10)", "-1.0"}, {"LBM_c1(10)", "0.0"}, {"LBM_c2(10)", "-1.0"},
		{"LBM_c0(11)", "0.0"}, {"LBM_c1(11)", "1.0"}, {"LBM_c2(11)", "1.0"},
		{"LBM_c0(12)", "0.0"}, {"LBM_c1(12)", "-1.0"}, {"LBM_c2(12)", "-1.0"},
		{"LBM_c0(13)", "1.0"}, {"LBM_c1(13)", "-1.0"}, {"LBM_c2(13)", "0.0"},
		{"LBM_c0(14)", "-1.0"}, {"LBM_c1(14)", "1.0"}, {"LBM_c2(14)", "0.0"},
		{"LBM_c0(15)", "1.0"}, {"LBM_c1(15)", "0.0"}, {"LBM_c2(15)", "-1.0"},
		{"LBM_c0(16)", "-1.0"}, {"LBM_c1(16)", "0.0"}, {"LBM_c2(16)", "1.0"},
		{"LBM_c0(17)", "0.0"}, {"LBM_c1(17)", "1.0"}, {"LBM_c2(17)", "-1.0"},
		{"LBM_c0(18)", "0.0"}, {"LBM_c1(18)", "-1.0"}, {"LBM_c2(18)", "1.0"},
		{"LBM_c0(19)", "1.0"}, {"LBM_c1(19)", "1.0"}, {"LBM_c2(19)", "1.0"},
		{"LBM_c0(20)", "-1.0"}, {"LBM_c1(20)", "-1.0"}, {"LBM_c2(20)", "-1.0"},
		{"LBM_c0(21)", "1.0"}, {"LBM_c1(21)", "1.0"}, {"LBM_c2(21)", "-1.0"},
		{"LBM_c0(22)", "-1.0"}, {"LBM_c1(22)", "-1.0"}, {"LBM_c2(22)", "1.0"},
		{"LBM_c0(23)", "1.0"}, {"LBM_c1(23)", "-1.0"}, {"LBM_c2(23)", "1.0"},
		{"LBM_c0(24)", "-1.0"}, {"LBM_c1(24)", "1.0"}, {"LBM_c2(24)", "-1.0"},
		{"LBM_c0(25)", "-1.0"}, {"LBM_c1(25)", "1.0"}, {"LBM_c2(25)", "1.0"},
		{"LBM_c0(26)", "1.0"}, {"LBM_c1(26)", "-1.0"}, {"LBM_c2(26)", "-1.0"},
		{"LBM_nbr(0,0,0)", "0"},
		{"LBM_nbr(1,0,0)", "1"},
		{"LBM_nbr(-1,0,0)", "2"},
		{"LBM_nbr(0,1,0)", "3"},
		{"LBM_nbr(0,-1,0)", "4"},
		{"LBM_nbr(0,0,1)", "5"},
		{"LBM_nbr(0,0,-1)", "6"},
		{"LBM_nbr(1,1,0)", "7"},
		{"LBM_nbr(-1,-1,0)", "8"},
		{"LBM_nbr(1,0,1)", "9"},
		{"LBM_nbr(-1,0,-1)", "10"},
		{"LBM_nbr(0,1,1)", "11"},
		{"LBM_nbr(0,-1,-1)", "12"},
		{"LBM_nbr(1,-1,0)", "13"},
		{"LBM_nbr(-1,1,0)", "14"},
		{"LBM_nbr(1,0,-1)", "15"},
		{"LBM_nbr(-1,0,1)", "16"},
		{"LBM_nbr(0,1,-1)", "17"},
		{"LBM_nbr(0,-1,1)", "18"},
		{"LBM_nbr(1,1,1)", "19"},
		{"LBM_nbr(-1,-1,-1)", "20"},
		{"LBM_nbr(1,1,-1)", "21"},
		{"LBM_nbr(-1,-1,1)", "22"},
		{"LBM_nbr(1,-1,1)", "23"},
		{"LBM_nbr(-1,1,-1)", "24"},
		{"LBM_nbr(-1,1,1)", "25"},
		{"LBM_nbr(1,-1,-1)", "26"},
		{"LBM_pexc(0)", "1"},
		{"LBM_pexc(1)", "3"},
		{"LBM_pexc(2)", "5"},
		{"LBM_pexc(3)", "7"},
		{"LBM_pexc(4)", "9"},
		{"LBM_pexc(5)", "11"},
		{"LBM_pexc(6)", "19"}
	}
};



std::string replace_BC_cond(std::string line)
{
	std::string new_line = line;
	std::string inner_line;
	
	
	size_t pos_curr_start = 0;
	size_t pos_curr_end = line.size()-1;
	pos_curr_start = line.rfind("LBM_COND_BC(");
	while (pos_curr_start != std::string::npos)
	{
		pos_curr_end = line.find(")END_LBM_COND_BC", pos_curr_start);
		inner_line = line.substr(pos_curr_start+12, pos_curr_end-(pos_curr_start+12));
		
		
		// Process arguments (needs six values of cX corresponding to pairs of cXi_p, cXi_nbr in the three directions).
		inner_line = replace_substring(inner_line, ",", " ");
		std::stringstream ss(inner_line);
		double cXi = 0;
		double cXni = 0;
		std::vector<std::string> cond_BC;
		for (int k = 0; k < 3; k++)
		{
			cXi = 0;
			cXni = 0;
			ss >> cXi >> cXni;
			if (cXni == 1.0)
				cond_BC.push_back("(" + coords[k] + "==Nbx-1)");
			else if (cXni == -1.0)
				cond_BC.push_back("(" + coords[k] + "==0)");
			else
			{
				if (cXi == 1.0)
					cond_BC.push_back("(" + coords[k] + "<Nbx-1)");
				else if (cXi == -1.0)
					cond_BC.push_back("(" + coords[k] + ">0)");
				else
					{}
			}
		}
		line = line.substr(0,pos_curr_start) + commatize(cond_BC,"and") + line.substr(pos_curr_start+(pos_curr_end-pos_curr_start+16), line.length());
		
		
		pos_curr_start = line.rfind("LBM_COND_BC(");
	}
	new_line = line;
	
	return new_line;
}

std::string replace_halo_combo(std::string line)
{
	std::string new_line = line;
	std::string inner_line;
	
	
	size_t pos_curr_start = 0;
	size_t pos_curr_end = line.size()-1;
	pos_curr_start = line.rfind("H_COMBO(");
	while (pos_curr_start != std::string::npos)
	{
		pos_curr_end = line.find(")END_H_COMBO", pos_curr_start);
		inner_line = line.substr(pos_curr_start+8, pos_curr_end-(pos_curr_start+8));
		
		
		// Process arguments (needs the number of dimensions and six values of cX corresponding to pairs of cXi_p, cXi_nbr in the three directions).
		inner_line = replace_substring(inner_line, ",", " ");
		std::stringstream ss(inner_line);
		double N_dim = 2;
		double cXi = 0;
		double cXni = 0;
		ss >> N_dim;
		std::vector<std::string> halo_combo;
		std::string incr = "";
		for (int k = 0; k < N_dim; k++)
		{
			cXi = 0;
			cXni = 0;
			ss >> cXi >> cXni;
			halo_combo.push_back("(" + coords[k] + "+1+(" + std::to_string((int)cXi) + ")+(" + std::to_string((int)cXni) + ")*Nbx)" + incr);
			incr = incr + "*(Nbx+2)";
		}
		line = line.substr(0,pos_curr_start) + commatize(halo_combo,"+") + line.substr(pos_curr_start+(pos_curr_end-pos_curr_start+12), line.length());
		
		
		pos_curr_start = line.rfind("H_COMBO(");
	}
	new_line = line;
	
	return new_line;
}

std::string replace_halo_limit(std::string line)
{
	std::string new_line = line;
	std::string inner_line;
	
	
	size_t pos_curr_start = 0;
	size_t pos_curr_end = line.size()-1;
	pos_curr_start = line.rfind("H_LIMIT(");
	while (pos_curr_start != std::string::npos)
	{
		pos_curr_end = line.find(")END_H_LIMIT", pos_curr_start);
		inner_line = line.substr(pos_curr_start+8, pos_curr_end-(pos_curr_start+8));
		
		
		// Process arguments (needs the number of dimensions and three coordinate components).
		inner_line = replace_substring(inner_line, ",", " ");
		std::stringstream ss(inner_line);
		int N_dim = 2;
		double cXi = 0;
		ss >> N_dim;
		std::vector<std::string> halo_limit;
		std::string incr = "";
		for (int k = 0; k < N_dim; k++)
		{
			cXi = 0;
			ss >> cXi;
			if (cXi == 1)
				halo_limit.push_back("(Nbx+1)"+incr);
			else
				halo_limit.push_back("(" + coords[k] + "+1)"+incr);
			incr = incr + "*(Nbx+2)";
		}
		line = line.substr(0,pos_curr_start) + commatize(halo_limit,"+") + line.substr(pos_curr_start+(pos_curr_end-pos_curr_start+12), line.length());
		
		
		pos_curr_start = line.rfind("H_LIMIT(");
	}
	new_line = line;
	
	return new_line;
}

std::string replace_halo_index(std::string line)
{
	std::string new_line = line;
	std::string inner_line;
	
	
	size_t pos_curr_start = 0;
	size_t pos_curr_end = line.size()-1;
	pos_curr_start = line.rfind("H_INDEX(");
	while (pos_curr_start != std::string::npos)
	{
		pos_curr_end = line.find(")END_H_INDEX", pos_curr_start);
		inner_line = line.substr(pos_curr_start+8, pos_curr_end-(pos_curr_start+8));
		
		
		// Process arguments (needs the number of dimensions and three coordinate components).
		inner_line = replace_substring(inner_line, ",", " ");
		std::stringstream ss(inner_line);
		int N_dim = 2;
		double cX = 0;
		double cY = 0;
		double cZ = 0;
		ss >> N_dim >> cX >> cY >> cZ;
		std::string halo_index = "(I_kap+1+" + std::to_string((int)cX) + ")+(J_kap+1+" + std::to_string((int)cY) + ")*(Nbx+2)";
		if (N_dim == 3)
			halo_index = halo_index + "+(K_kap+1+" + std::to_string((int)cZ) + ")*(Nbx+2)*(Nbx+2)";
		line = line.substr(0,pos_curr_start) + halo_index + line.substr(pos_curr_start+(pos_curr_end-pos_curr_start+12), line.length());
		
		
		pos_curr_start = line.rfind("H_INDEX(");
	}
	new_line = line;
	
	return new_line;
}






std::string FormatLinePre_LBM(std::string line)
{
	std::string new_line = line;
	//line = replace_BC_cond(line);
	
	return new_line;
}

std::string FormatLinePost_LBM(std::string s_output)
{
	std::string new_output = s_output;
	new_output = replace_BC_cond(new_output);
	new_output = replace_halo_index(new_output);
	new_output = replace_halo_limit(new_output);
	new_output = replace_halo_combo(new_output);
	
	return new_output;
}
