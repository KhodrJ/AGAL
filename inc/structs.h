/**************************************************************************************/
/*                                                                                    */
/*  Author: Khodr Jaber                                                               */
/*  Affiliation: Turbulence Research Lab, University of Toronto                       */
/*                                                                                    */
/**************************************************************************************/

#ifndef STRUCTS_H
#define STRUCTS_H

struct is_marked_for_refinement
{
    __device__ bool operator()(const int ref_id)
    {
        return ref_id==V_REF_ID_MARK_REFINE;
    }
};

struct is_marked_for_coarsening
{
    __device__ bool operator()(const int ref_id)
    {
        return ref_id==V_REF_ID_MARK_COARSEN;
    }
};

struct is_marked_for_removal
{
    __device__ bool operator()(const int ref_id)
    {
        return ref_id==V_REF_ID_REMOVE;
    }
};

struct is_newly_generated
{
    __device__ bool operator()(const int ref_id)
    {
        return ref_id==V_REF_ID_NEW;
    }
};

struct is_equal_to
{
    is_equal_to(int level) : level_{level} {}
    __device__ bool operator()(const int level)
    {
        return level==level_;
    }
    int level_;
};

struct is_not_equal_to
{
    is_not_equal_to(int level) : level_{level} {}
    __device__ bool operator()(const int level)
    {
        return level!=level_;
    }
    int level_;
};

struct is_equal_to_zip
{
    int val;
    is_equal_to_zip(int v) : val(v) {}
    __device__ bool operator()(const thrust::tuple<int, int>& t) const  // adjust types accordingly
    {
        return thrust::get<0>(t) == val;  // only compares the first zipped element
    }
};

struct is_removed
{
    __device__ bool operator()(const int ID)
    {
        return ID==N_SKIPID;
    }
};

struct is_not_removed
{
    __device__ bool operator()(const int ID)
    {
        return ID!=N_SKIPID;
    }
};

struct is_nonnegative
{
    __device__ bool operator()(const int ID)
    {
        return ID>=0;
    }
};

struct is_positive
{
    __device__ bool operator()(const int ID)
    {
        return ID>0;
    }
};

struct is_nonnegative_and_less_than
{
    is_nonnegative_and_less_than(int ID) : val_{ID} {}
    __device__ bool operator()(const int ID) const
    {
        return ID>=0 && ID<val_;
    }
    int val_;
};

struct is_positive_and_less_than
{
    is_positive_and_less_than(int ID) : val_{ID} {}
    __device__ bool operator()(const int ID) const
    {
        return ID>0 && ID<val_;
    }
    int val_;
};

struct replace_diff_with_indexM1
{
    __device__ int operator()(const int& index, const int& val) const
    {
        return (val > 0) ? index-1 : val;
    }
};

#endif
