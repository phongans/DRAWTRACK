// dllmain.cpp : Defines the entry point for the DLL application.
#include "pch.h"
#include "dllocsort.h"

BOOL APIENTRY DllMain( HMODULE hModule,
                       DWORD  ul_reason_for_call,
                       LPVOID lpReserved
                     )
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}

/**
@brief Convert Vector to Matrix
@param data
@return Eigen::Matrix<float, Eigen::Dynamic, 6>
*/
Eigen::Matrix<float, Eigen::Dynamic, 6> Vector2Matrix(std::vector<Eigen::RowVectorXf> data) {
    Eigen::Matrix<float, Eigen::Dynamic, 6> matrix(data.size(), data[0].size());
    for (int i = 0; i < data.size(); ++i) {
        for (int j = 0; j < data[0].size(); ++j) {
            matrix(i, j) = data[i][j];
        }
    }
    return matrix;
}

extern "C" __declspec (dllexport) int __cdecl CreateTracker(ocsort::OCSort** ocSort, float det_thresh_, int max_age_, int min_hits_, float iou_threshold_,
    int delta_t_, const char* asso_func_, float inertia_, bool use_byte_) {
    *ocSort = new ocsort::OCSort(det_thresh_, max_age_, min_hits_, iou_threshold_, delta_t_, asso_func_, inertia_, use_byte_);
    if (*ocSort == nullptr) {
        return 0;
    }
    return 1;
}


extern "C" __declspec (dllexport) void __cdecl UpdateTracker(ocsort::OCSort* ocSort, std::vector<Eigen::RowVectorXf>& data) {
    data = ocSort->update(Vector2Matrix(data));
}