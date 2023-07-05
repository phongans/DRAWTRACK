// dllmain.cpp : Defines the entry point for the DLL application.
#include "pch.h"
#include "dllocsort.h"
#include <nlohmann/json.hpp>
#pragma warning(disable: 4996)

using json = nlohmann::json;

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

template<typename AnyCls>
std::ostream& operator<<(std::ostream& os, const std::vector<AnyCls>& v) {
    os << "{";
    for (auto it = v.begin(); it != v.end(); ++it) {
        os << "(" << *it << ")";
        if (it != v.end() - 1) os << ", ";
    }
    os << "}";
    return os;
}

/**
@brief Convert Vector to Matrix
@param data
@return Eigen::Matrix<float, Eigen::Dynamic, 6>
*/
Eigen::Matrix<float, Eigen::Dynamic, 6> Vector2Matrix(std::vector<std::vector<float>> data) {
    Eigen::Matrix<float, Eigen::Dynamic, 6> matrix(data.size(), data[0].size());
    for (int i = 0; i < data.size(); ++i) {
        for (int j = 0; j < data[0].size(); ++j) {
            matrix(i, j) = data[i][j];
        }
    }
    return matrix;
}

/**
@brief Convert VectorOfEigenRowVectorXf to JsonArray
@param vectorOfVectors
@return array e.g:[[1062.0,570.0,1219.0,855.0,4.0,0.0,0.8661319017410278]]
*/
json VectorOfEigenRowVectorXfToJson(const std::vector<Eigen::RowVectorXf> vectorOfVectors) {
    json jsonArray;
    for (const auto& rowVector : vectorOfVectors) {
        json jsonObject;
        for (int i = 0; i < rowVector.size(); ++i) {
            jsonObject.push_back(rowVector(i));
        }
        jsonArray.push_back(jsonObject);
    }
    return jsonArray;
}


extern "C" __declspec (dllexport) int __cdecl CreateTracker(ocsort::OCSort** ocSort, float det_thresh_, int max_age_, int min_hits_, float iou_threshold_,
    int delta_t_, const char* asso_func_, float inertia_, bool use_byte_) {
    *ocSort = new ocsort::OCSort(det_thresh_, max_age_, min_hits_, iou_threshold_, delta_t_, asso_func_, inertia_, use_byte_);
    if (*ocSort == nullptr) {
        return 0;
    }
    return 1;
}


extern "C" __declspec (dllexport) const char* __cdecl UpdateTracker(ocsort::OCSort* ocSort, std::vector<std::vector<float>>& data) { 
    std::vector<Eigen::RowVectorXf> res = ocSort->update(Vector2Matrix(data));

    json obj_tracker = VectorOfEigenRowVectorXfToJson(res);

    std::string json_str = obj_tracker.dump();
    char* c_str = new char[json_str.length() + 1];
    std::strcpy(c_str, json_str.c_str());

    return c_str;
}