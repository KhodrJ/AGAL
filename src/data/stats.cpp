/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

// This script takes the output of the complex geometry paper and extracts the
// statistics on the execution times across several runs.

#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <numeric>
#include <algorithm>
#include <math.h>
#include <boost/math/distributions/students_t.hpp>

// From ChatGPT.
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        os << vec[i];
        if (i != vec.size() - 1)
            os << ", ";
    }
    os << "]";
    return os;
}

// Compute the mean
template <typename T>
double ComputeMean(const std::vector<T>& data) {
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    return sum / static_cast<double>( data.size() );
}

// Compute the standard deviation (sample)
template <typename T>
double ComputeStandardDeviation(const std::vector<T>& data, double mean) {
    double sumSquaredDiffs = 0.0;
    for (T value : data) {
        sumSquaredDiffs += (static_cast<double>(value) - mean) * (static_cast<double>(value) - mean);
    }
    return std::sqrt(sumSquaredDiffs / (static_cast<double>(data.size()) - 1.0));
}

double GetTCriticalValue(double confidence_level, int sample_size) {
    if (sample_size < 2)
        throw std::invalid_argument("Sample size must be at least 2.");

    int degrees_of_freedom = sample_size - 1;

    // Create the t-distribution with the given degrees of freedom
    boost::math::students_t_distribution<double> t_dist(degrees_of_freedom);

    // For two-tailed CI, use (1 + confidence_level) / 2 (e.g., 0.975 for 95% CI)
    double cumulative_prob = (1.0 + confidence_level) / 2.0;

    // Get the t-critical value (quantile)
    double t_critical = boost::math::quantile(t_dist, cumulative_prob);
    return t_critical;
}

// Compute 95% confidence interval (returns lower and upper bounds)
template <typename T>
std::pair<double, double> ComputeConfidenceInterval95(const std::vector<T>& data) {
    if (data.size() < 2) {
        throw std::invalid_argument("Need at least two data points for confidence interval");
    }

    double mean = ComputeMean(data);
    double stddev = ComputeStandardDeviation(data, mean);
    double n = data.size();
    double marginOfError;

    // Use t-distribution critical value for small samples (n <= 30)
    // You can replace this with a lookup or use boost/math for exact t*
    //double t_value = (n > 30) ? 1.96 : 2.045; // rough t* for df ~ 20 at 95% CI
    double t_value = GetTCriticalValue(0.95,static_cast<int>(n));
    
    marginOfError = t_value * (stddev / std::sqrt(n));

    return {mean - marginOfError, mean + marginOfError};
}

std::string GetFileName(int iter, std::string ver)
{
    std::string fn = "ERR";
    
    if (ver == "BINS_N")
        fn = "out_n_" + std::to_string(iter) + ".txt";
    if (ver == "BINS_R")
        fn = "out_r_" + std::to_string(iter) + ".txt";
    if (ver == "BINS_C")
        fn = "out_c_" + std::to_string(iter) + ".txt";
    if (ver == "BINS_RC")
        fn = "out_rc_" + std::to_string(iter) + ".txt";
    if (ver == "BINS_RMC")
        fn = "out_rmc_" + std::to_string(iter) + ".txt";
    
    return fn;
}

void ExtractTimeFrom(std::string s, std::string line, std::map<std::string, std::vector<int>> &map)
{
    size_t pos = line.rfind(s);
    if (pos != std::string::npos)
    {
        std::string line_substr = line.substr(pos+s.size(),line.size());
        std::stringstream ss(line_substr);
        
        int t;
        ss >> t;
        map[s].push_back(t);
    }
}

std::vector<int> ExtractLevelData(const std::vector<int> &data, const int &l, const int &Nl)
{
    std::vector<int> data_l;
    int Nt = data.size();
    int N = Nt/Nl;
    
    // Assumes that the data has a size of exactly N*Nl.
    for (int i = 0; i < N; i++)
        data_l.push_back( data[l + Nl*i] );
    
    return data_l;
}

int main(int argc, char *argv[])
{
    if (argc > 3)
    {
        int Nruns = std::stoi(argv[1]);
        int Nlevels = std::stoi(argv[2]);
        std::string ver = std::string(argv[3]);
        std::map<std::string, std::vector<int>> map;
        
        
        
        const std::vector<std::string> TimeNames = 
        {
            "MemoryAllocation(1)3D:",
            "MemoryAllocation(2)3D:",
            "ComputingRayIndicators1D:",
            "GatheredRayIndicators1D:",
            "ScatteredRayIndicators1D:",
            "ComputingBoundingBoxLimits3D:",
            "Compaction3D:",
            "SortByKey3D:",
            "ReductionByKey3D:",
            "Scatter(1)3D:",
            "AdjacentDifference3D:",
            "CopyIf3D:",
            "Scatter(2)3D:",
            "MemoryAllocation(1)2D:",
            "ComputingBoundingBoxLimits2D:",
            "Compaction2D:",
            "SortByKey2D:",
            "ReductionByKey2D:",
            "Scatter(1)2D:",
            "AdjacentDifference2D:",
            "CopyIf2D:",
            "Scatter(2)2D:",
            
            
            
            "MemoryAllocation(1)MD:",
            "MemoryAllocation(2)MD:",
            "ComputingRayIndicatorsMD:",
            "GatheredRayIndicatorsMD:",
            "ScatteredRayIndicatorsMD:",
            "ComputingBoundingBoxLimitsMD:",
            "CompactionMD:",
            "SortByKeyMD:",
            "ReductionByKeyMD:",
            "Scatter(1)MD:",
            "AdjacentDifferenceMD:",
            "CopyIfMD:",
            "Scatter(2)MD:",
            
            
            
            
            
            
            "Voxelize:",
            "VoxelizePropagate(+x):",
            "VoxelizePropagate(-x):",
            "VoxelizeUpdateMasks:",
            "MarkBoundary:",
            "MarkExterior:",
            "Finalize:",
            "UpdateSolidChildren:",
            "CheckMasks(1):",
            "CheckMasks(2):",
            
            
            
            "LinkLengthComputation:",
            "LinkLengthValidation:"
        };
        
        
        
        for (int k = 0; k < Nruns; k++)
        {
            std::cout << "Processing run " << k << std::endl;
            
            // Get the file name.
            std::string filename = GetFileName(k, ver);
            if (filename == "ERR")
            {
                std::cout << "Invalid filename..." << std::endl;
                return 0;
            }
            
            // Open appropriate file.
            std::ifstream in;
            if (in = std::ifstream(filename))
            {
                // Loop over lines in the file.
                for (std::string line; std::getline(in,line);)
                {
                    std::stringstream ss(line);
                    
                    for (int p = 0; p < TimeNames.size(); p++)
                        ExtractTimeFrom(TimeNames[p], line, map);
                }
            }
            else
            {
                std::cout << "Could not open file (" << filename << ")..." << std::endl;
            }
        }
        
        // Now check the output.
        for (int p = 0; p < TimeNames.size(); p++)
        {
            //std::cout << TimeNames[p] << " " << map[TimeNames[p]] << std::endl;
            std::string name = TimeNames[p];
            std::vector<int> data = map[name];
            //std::cout << name << " | " << data << std::endl;
            for (int l = 0; l < Nlevels; l++)
            {
                std::vector<int> data_l = ExtractLevelData(data,l,Nlevels);
                if (data_l.size() > 2)
                {
                    //std::cout << name << "_" << l  << " | " << data_l << std::endl;
                    std::pair<double,double> pair_l = ComputeConfidenceInterval95(data_l);
                    double mean_l = ComputeMean(data_l);
                    std::cout << name << "_" << l  << " | [" << pair_l.first << ", " << mean_l << ", " << pair_l.second << "]   sym=[" << pair_l.second-mean_l << "," << mean_l-pair_l.first << "]" << std::endl;
                }
            }
            
        }
    }
    else
    {
        std::cout << "Insufficient number of arguments..." << std::endl;
        return 0;
    }
}
