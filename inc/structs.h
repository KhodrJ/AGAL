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

#endif
