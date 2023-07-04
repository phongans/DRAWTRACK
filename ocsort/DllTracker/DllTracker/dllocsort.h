#pragma once
#ifndef DLL_OCSORT_H
#define DLL_OCSORT_H

#include <OCSort.hpp>

extern "C" __declspec (dllexport) int __cdecl CreateTracker(ocsort::OCSort** ocSort, float det_thresh_, int max_age_ = 30, int min_hits_ = 3, float iou_threshold_ = 0.3,
	int delta_t_ = 3, const char* asso_func_ = "iou", float inertia_ = 0.2, bool use_byte_ = false);

extern "C" __declspec (dllexport) void __cdecl UpdateTracker(ocsort::OCSort* ocSort, std::vector<Eigen::RowVectorXf>& data);

#endif // DLL_OCSORT_H
