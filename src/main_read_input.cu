#include "cppspec.h"

int ParseNxFactor(std::string str_val, int Nx)
{
    // Returns the result of A*Nx where A is the specified integer. Nx must come second (so not Nx*A), and 1*Nx should be written explicitly (not just Nx).
    int pos = str_val.find("*");
    int A = std::stoi(str_val.substr(0, pos));
    
    if (pos != std::string::npos)
        return A*Nx;
    else
        return A;
}

double ParseNxFactor_D(std::string str_val, int Nx)
{
    // Returns the result of A*Nx where A is the specified integer. Nx must come second (so not Nx*A), and 1*Nx should be written explicitly (not just Nx).
    int pos = str_val.find("*");
    double A = std::stod(str_val.substr(0, pos));
    
    if (pos != std::string::npos)
        return A*Nx;
    else
        return A;
}

int ReadInputFile(std::string input_file_directory, std::map<std::string, int> &input_map_int, std::map<std::string, double> &input_map_dbl, std::map<std::string, std::string> &input_map_str)
{
    // Setup default input map.
    std::ifstream input = std::ifstream(input_file_directory + "input.txt");
    input_map_int = std::map<std::string, int> // Define map of integer parameters with default values.
    {
        {"MAX_LEVELS_WALL",         5},
        {"MAX_LEVELS_INTERIOR",     4},
        {"N_ITER_TOTAL",            10*64},
        {"N_LEVEL_START",           0},
        {"Nx",                      64},
        {"N_RESTART",               1},
        {"G_LOADTYPE",              0},
        {"G_PRINT",                 1},
        {"G_BIN_OVERLAP",           10},
        {"G_BIN_DENSITY",           1},
        {"G_BIN_FRAC",              1},
        {"G_BIN_APPROACH",          0},
        {"G_BIN_LEVELS",            1},
        {"PERIODIC_X",              0},
        {"PERIODIC_Y",              0},
        {"PERIODIC_Z",              0},
        {"S_COLLISION",             0},
        {"S_INTERP",                1},
        {"S_INTERP_HYBRID",         0},
        {"S_AVERAGE",               0},
        {"S_CRITERION",             0},
        {"S_LES",                   0},
        {"S_FORCE_ORDER",           0},
        {"S_FORCE_TYPE",            0},
        {"S_BC_TYPE",               0},
        {"P_REFINE",                32},
        {"N_PROBE",                 0},
        {"N_PROBE_DENSITY",         4},
        {"N_PROBE_FREQUENCY",       32},
        {"N_PROBE_FORCE",           0},
        {"N_PROBE_F_START",         0},
        {"N_PROBE_F_FREQUENCY",     16},
        {"N_PROBE_AVE",             0},
        {"N_PROBE_AVE_FREQUENCY",   1},
        {"N_PROBE_AVE_START",       100*64},
        {"N_PRINT_LEVELS",          2},
        {"N_PRINT_LEVELS_LEGACY",   2},
        {"N_PRINT_LEVELS_IMAGE",    2},
        {"P_OUTPUT",                1*64},
        {"N_OUTPUT",                10},
        {"N_OUTPUT_START",          0},
        {"VOL_I_MIN",               0},
        {"VOL_I_MAX",               64},
        {"VOL_J_MIN",               0},
        {"VOL_J_MAX",               64},
        {"VOL_K_MIN",               0},
        {"VOL_K_MAX",               64},
        {"USE_CPU",                 1}
    };
    input_map_dbl = std::map<std::string, double> // Define map of double parameters with default values.
    {
        {"L_c",                     1.0},
        {"L_fy",                    1.0},
        {"L_fz",                    1.0},
        {"v0",                      1.5625e-5},
        {"G_NEAR_WALL_DISTANCE",    0.05},
        {"s1",                      1.19},
        {"s2",                      1.40},
        {"s3",                      1.20},
        {"s4",                      1.20},
        {"s5",                      1.20},
        {"s6",                      1.98},
        {"V_PROBE_TOL",             1e-4},
        {"N_REFINE_START",          -3.0},
        {"N_REFINE_INC",            1.0},
        {"N_REFINE_MAX",            1.0},
        {"S_FORCEVOLUME_Xm",        0.0},
        {"S_FORCEVOLUME_XM",        0.0},
        {"S_FORCEVOLUME_Ym",        0.0},
        {"S_FORCEVOLUME_YM",        0.0},
        {"S_FORCEVOLUME_Zm",        0.0},
        {"S_FORCEVOLUME_ZM",        0.0},
    };
    input_map_str = std::map<std::string, std::string> // Define map of string parameters with default values.
    {
        {"I_DIR_NAME",              input_file_directory},
        {"P_DIR_NAME",              "../out/TEST_UPLOAD/"},
        {"G_FILENAME",              "bunny_ascii.stl"}
    };
    
    // Extract Nx first.
    int Nx = -1;
    std::string str_name;
    std::string str_val;
    for (std::string line; std::getline(input, line);) 
    {
        if (line.length() > 0 && line.at(0) != '#') // Ignore comments.
        {
            std::stringstream ss(line);
            ss >> str_name >> str_val;
            
            if (str_name == "Nx")
                Nx = std::stoi(str_val);
        }
    }
    
    // Read rest of input file.
    input.clear();
    input.seekg(0);
    for (std::string line; std::getline(input, line);) 
    {
        if (line.length() > 0 && line.at(0) != '#') // Ignore comments.
        {
            std::stringstream ss(line);
            ss >> str_name >> str_val;
            if ( input_map_int.count(str_name) == 1 )
                input_map_int[str_name] = ParseNxFactor(str_val, Nx);
            else if ( input_map_dbl.count(str_name) == 1 )
                input_map_dbl[str_name] = std::stod(str_val);
            else if ( input_map_str.count(str_name) == 1 )
                input_map_str[str_name] = str_val;
            else
                std::cout << "Error: Invalid input (" << str_name << ") skipped..." << std::endl;
        }
    }
    
    // Check (debug).
//     for (std::map<std::string, int>::iterator it = input_map_int.begin(); it != input_map_int.end(); ++it)
//     {
//         std::cout << it->first << " " << it->second << std::endl;
//     }
//     for (std::map<std::string, double>::iterator it = input_map_dbl.begin(); it != input_map_dbl.end(); ++it)
//     {
//         std::cout << it->first << " " << it->second << std::endl;
//     }
    
    return 0;
}
