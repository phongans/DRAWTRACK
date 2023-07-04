#include "src/inference_trt.h"

template<typename Object>
std::ostream& operator<<(std::ostream& os, const std::vector<Object>& v) {
    os << "{";
    for (auto it = v.begin(); it != v.end(); ++it) {
        os << "(" << *it << ")";
        if (it != v.end() - 1) os << ", ";
    }
    os << "}";
    return os;
}

// Runs object detection on video stream then displays annotated results.
int main(int argc, char* argv[]) {
    // Parse the command line arguments
    // Must pass the model path as a command line argument to the executable
    if (argc < 2) {
        std::cout << "Error: Must specify the model path" << std::endl;
        std::cout << "Usage: " << argv[0] << "/path/to/onnx/model.onnx" << std::endl;
        return -1;
    }

    if (argc > 3) {
        std::cout << "Error: Too many arguments provided" << std::endl;
        std::cout << "Usage: " << argv[0] << "/path/to/onnx/model.onnx" << std::endl;
    }

    // Ensure the onnx model exists
    const std::string onnxModelPath = argv[1];
    if (!doesFileExist(onnxModelPath)) {
        std::cout << "Error: Unable to find file at path: " << onnxModelPath << std::endl;
        return -1;
    }

    InferenceTRT inference(onnxModelPath);

    // Initialize the video stream
    cv::VideoCapture capture("C:\\Projects\\Research\\Models\\road.mp4");

    while (true) {
        // Grab frame
        cv::Mat frame;

        if (!capture.read(frame)) {
            std::cout << "\n Cannot read the video file. Please check your video.\n";
            break;
        }

        // Run inference
        auto objects = inference.detectObjects(frame);

        // Draw the bounding boxes on the image
        inference.drawObjectLabels(frame, objects);

        // Display the results
        cv::imshow("Object Detection TRT Verify", frame);
        if (cv::waitKey(1) >= 0) {
            break;
        }
    }
    return 0;
}