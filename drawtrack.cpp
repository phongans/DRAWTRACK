#include "detector/include/inference_trt.h"
#include "ocsort/include/OCSort.hpp"
#include <Eigen/Dense>
#include <thread>

std::mutex dataMutex;

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

void processFrame(const cv::Mat& frame, std::vector<Object>& objects, ocsort::OCSort& tracker, InferenceTRT& inference) {
    std::vector<std::vector<float>> data;
    for (auto const& value : objects) {
        std::vector<float> row;
        for (;;) {
            row.push_back(value.rect.x);
            row.push_back(value.rect.y);
            row.push_back(value.rect.x + value.rect.width);
            row.push_back(value.rect.y + value.rect.height);
            row.push_back(value.probability);
            row.push_back(value.label);
            break;
        }
        data.push_back(row);

        std::vector<Eigen::RowVectorXf> res = tracker.update(Vector2Matrix(data));

        for (auto j : res) {
            int ID = int(j[4]);
            int Class = int(j[5]);
            float conf = j[6];
            cv::putText(frame, cv::format("ID:%d", ID), cv::Point(j[0], j[1] - 5), 0, 2, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
            cv::rectangle(frame, cv::Rect(j[0], j[1], j[2] - j[0] + 1, j[3] - j[1] + 1), cv::Scalar(0, 0, 255), 1);
        }

        data.clear();
    }
}

void videoODThread(const std::string& modelfile) {

    InferenceTRT inference = InferenceTRT(modelfile);

    ocsort::OCSort tracker = ocsort::OCSort(0, 50, 1, 0.22136877277096445, 1, "giou", 0.3941737016672115, true);

    cv::VideoCapture capture("C:\\Projects\\Research\\Models\\road.mp4");

    bool isRunning = true;  // Variable for controlling the loop

    while (isRunning) {
        cv::Mat frame;
        if (!capture.read(frame)) {
            std::cout << "\n Cannot read the video file. Please check your video.\n";
            isRunning = false;
            break;
        }

        auto objects = inference.detectObjects(frame);

        std::lock_guard<std::mutex> lock(dataMutex);

        processFrame(frame, objects, tracker, inference);

        cv::imshow("VideoOD", frame);

        if (cv::waitKey(1) == 27)
            isRunning = false;
    }
    
}

int main() {
    std::string modelfile = "C:/Projects/Research/Models/yolov8s.onnx";
    std::thread videoThread(videoODThread, modelfile);
    videoThread.join();
    std::cout << "End of program." << std::endl;
    return 0;
}