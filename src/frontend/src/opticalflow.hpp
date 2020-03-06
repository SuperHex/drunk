#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <queue>
#include <vector>
#include <memory>

class lazyOpticalFlow {
    lazyOpticalFlow(const std::string& file, size_t cacheSize) { loadVideo(file); }

    void loadVideo(const std::string& file) {
        _capture = cv::VideoCapture{cv::samples::findFile(file)};
        if (!_capture.isOpened()) {
            std::cout << "Error open file: " << file << std::endl;
        }
    }

    cv::Mat& operator()() {
        if (!_flows.empty()) { _flows.pop(); }
        size_t loop = _flows.empty() ? _cacheSize : 1;

        cv::Mat tmp, frame1, frame2;
        _capture >> tmp;
        cv::cvtColor(tmp, frame1, cv::COLOR_BGR2GRAY);
        for (auto i = 0; i < loop; i++) {
            _capture >> tmp;
            cv::cvtColor(tmp, frame2, cv::COLOR_BGR2GRAY);
            cv::Mat flow(frame1.size(), CV_32FC2);
            cv::calcOpticalFlowFarneback(frame1, frame2, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
            _flows.emplace(std::move(flow));
        }

        cv::Mat& front = _flows.front();
        return front;
    }

private:
    size_t _cacheSize;
    cv::VideoCapture _capture;
    std::queue<cv::Mat> _flows;
};