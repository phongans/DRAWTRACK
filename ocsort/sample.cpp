#include <OCSort.hpp>
#include <Eigen/Dense>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
#include <cassert>

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

Eigen::Matrix<float, Eigen::Dynamic, 6> read_csv_to_eigen(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file!" << std::endl;
        exit(1);
    }
    std::string line;
    std::vector<std::vector<float>> data;
    while (std::getline(file, line)) {
        std::vector<float> row;
        std::istringstream iss(line);
        std::string field;
        while (std::getline(iss, field, ',')) {
            row.push_back(std::stof(field));
        }
        data.push_back(row);
    }

    Eigen::Matrix<float, Eigen::Dynamic, 6> matrix(data.size(), data[0].size());
    for (int i = 0; i < data.size(); ++i) {
        for (int j = 0; j < data[0].size(); ++j) {
            matrix(i, j) = data[i][j];
        }
    }
    return matrix;
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

std::vector<float> String2Vector(std::string line, bool Format = true) {
    std::vector<float> x;
    std::stringstream ss(line);
    std::string item;
    std::vector<float> data;
    while (std::getline(ss, item, ',')) {
        data.push_back(std::stod(item));
    }
    // After retrieving the data from this line, we need to format it.
    float x1 = data[2];
    float y1 = data[3];
    float x2 = data[2] + data[4];
    float y2 = data[3] + data[5];
    // The desired data format is: xyxys, score, 0
    x.insert(x.end(), { x1, y1, x2, y2, data[6], 0 });
    if (Format)
        return x;
    else
        return data;
}

int main(int argc, char* argv[]) {
    std::ostringstream filename;
    filename << "C:/Projects/Research/Models/MOT17-02.txt";
    std::ifstream filetxt(filename.str());

    if (filetxt.is_open()) {
        std::cout << "File is Opened, Path is:" << filename.str() << "\n";
        std::string line;
        int flag = 1;
        std::vector<Eigen::MatrixXf> ALL_INPUT;
        Eigen::MatrixXf FRAME_INPUT;
        std::vector<std::vector<float>> frame;
        std::vector<float> frame_previous;
        std::vector<float> tmp_fmt;
        std::vector<float> tmp;
        while (true) {
            std::getline(filetxt, line);
            if (filetxt.eof()) {
                std::cout << "Read the END OF FILES" << std::endl;
                if (frame_previous.size() != 0)
                    frame.push_back(frame_previous);
                ALL_INPUT.push_back(Vector2Matrix(frame));
                break;
            }
            tmp_fmt = String2Vector(line);
            tmp = String2Vector(line, false);
            // std::cout << tmp[0] << std::endl;
            if ((int)tmp[0] != flag) {
                flag = (int)tmp[0];
                frame_previous = tmp_fmt;
                FRAME_INPUT = Vector2Matrix(frame);
                ALL_INPUT.push_back(FRAME_INPUT);
                // std::cout << frame << std::endl;
                frame.clear();
            }
            else {
                if (frame_previous.size() != 0) {
                    frame.push_back(frame_previous);
                    frame_previous.clear();
                }
                frame.push_back(tmp_fmt);
            }
        }

        for (const auto& element : frame) {
            std::cout << element << std::endl;
        }

        // At this point, all frame data has been stored in the ALL_INPUT variable.
        std::cout << "Size of data:" << ALL_INPUT.size() << std::endl;
        ocsort::OCSort A = ocsort::OCSort(0, 50, 1, 0.22136877277096445, 1, "giou", 0.3941737016672115, true);
        float OverAll_Time = 0.0;
        // Iterate through all the inputs and pass them to OCSORT
        for (auto dets : ALL_INPUT) {
            auto T_start = high_resolution_clock::now();
            std::vector<Eigen::RowVectorXf> res = A.update(dets);
            auto T_end = high_resolution_clock::now();
            duration<float, std::milli> ms_float = T_end - T_start;
            OverAll_Time += ms_float.count();
        }
        // Calculate the average frame rate.
        float avg_cost = OverAll_Time / ALL_INPUT.size();
        int FPS = int(1000 / avg_cost);
        std::cout << "Average Time Cost: " << avg_cost << " Avg FPS: " << FPS << std::endl;
    }
    else {
        std::cout << "open Failed" << std::endl;
    }
    return 0;
}