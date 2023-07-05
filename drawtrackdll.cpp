#include "inference_trt.h"
#include "dllocsort.h"
#include <Eigen/Dense>
#include <thread>
#include <nlohmann/json.hpp>

std::mutex dataMutex;

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

void processFrame(const cv::Mat& frame, std::vector<Object>& objects, ocsort::OCSort* tracker, InferenceTRT& inference) {
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

        std::string jsonStr = UpdateTracker(tracker, data);
        
        nlohmann::json jsonData = nlohmann::json::parse(jsonStr);

        for (const auto& rect : jsonData) {
            // Rectangle bbox tracker
            double x1 = rect[0];
            double y1 = rect[1];
            double x2 = rect[2];
            double y2 = rect[3];
            double id = rect[4];
            double label = rect[5];
            double conf = rect[6];

            cv::putText(frame, cv::format("ID:%.0f", id) + cv::format("(%.2f)", conf), cv::Point(x1, y1 - 5), 0, 1, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
            cv::rectangle(frame, cv::Rect(x1, y1, x2 - x1 + 1, y2 - y1 + 1), cv::Scalar(0, 0, 255), 1);
        }
        data.clear();
    }
}

void videoODThread(const std::string& modelfile) {

    InferenceTRT inference = InferenceTRT(modelfile);

    ocsort::OCSort* tracker;
    CreateTracker(&tracker, 0, 50, 1, 0.22136877277096445, 1, "giou", 0.3941737016672115, true);

    cv::VideoCapture capture("../../data/video.mp4");

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
    std::string modelfile = "../../data/yolov8s.onnx";
    std::thread videoThread(videoODThread, modelfile);
    videoThread.join();
    std::cout << "End of program." << std::endl;
    return 0;
}