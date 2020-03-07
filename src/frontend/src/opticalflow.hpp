#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <torch/script.h>

#include <chrono>
#include <deque>
#include <vector>

#ifndef __clang__
#include <execution>
#endif

torch::Tensor matToTensor(const cv::Mat& mat) {
    auto tensor = torch::from_blob(mat.data, {1, mat.rows, mat.cols, 1}, at::kByte);
    tensor = tensor.permute({{0, 3, 1, 2}});
    return tensor;
}

class opticalflow {

public:

    opticalflow(const std::string& file) { loadVideo(file); }

    void loadVideo(const std::string& file) {
        capture = cv::VideoCapture{cv::samples::findFile(file)};
        if (!capture.isOpened()) {
            std::cout << "Error open file: " << file << std::endl;
        }
    }

    size_t preComputeAll() {
        std::vector<cv::Mat> buffer;

        // assume at least one frame
        auto startTime = std::chrono::steady_clock::now();
        cv::Mat tmp, tmp2, frame;
        while (true) {
            capture >> tmp;

            if (tmp.empty()) {
                break;
            }

            cv::resize(tmp, tmp2, cv::Size(224, 224), 0, 0, cv::INTER_AREA);
            cv::cvtColor(tmp2, frame, cv::COLOR_BGR2GRAY);
            buffer.emplace_back(std::move(frame));
        }
        auto endTime = std::chrono::steady_clock::now();
        std::chrono::duration<double> diff = endTime - startTime;
        std::cout << "Transform all images in " << diff.count() << " s" << std::endl;

        std::vector<cv::Mat> flow_buffer;
        flow_buffer.reserve(buffer.size() - 1);
        startTime = std::chrono::steady_clock::now();
        #if defined(__clang__)
        std::transform(buffer.begin(), buffer.end() - 1, buffer.begin() + 1, std::back_inserter(flow_buffer),
            [](const auto& frame1, const auto& frame2) {
                cv::Mat flow(frame1.size(), CV_32FC2);
                cv::calcOpticalFlowFarneback(frame1, frame2, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
                return flow;
            } );
        std::for_each(flow_buffer.begin(), flow_buffer.end(), [&](const auto& flow) {
            cv::Mat parts[2];
            cv::split(flow, parts);
            flows_x.emplace_back(std::move(parts[0]));
            flows_y.emplace_back(std::move(parts[1]));
        });
        #else
        std::transform(std::execution::par, buffer.begin(), buffer.end() - 1, buffer.begin() + 1, std::back_inserter(flows),
            [](const auto& frame1, const auto& frame2) {
                cv::Mat flow(frame1.size(), CV_32FC2);
                cv::calcOpticalFlowFarneback(frame1, frame2, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
                return flow;
            } );
        std::for_each(std::execution::par_seq, flow_buffer.begin(), flow_buffer.end(), [&](const auto& flow) {
            cv::Mat parts[2];
            cv::split(flow, parts);
            flows_x.emplace_back(std::move(parts[0]));
            flows_y.emplace_back(std::move(parts[1]));
        });
        #endif
        endTime = std::chrono::steady_clock::now();
        diff = endTime - startTime;
        std::cout << "Precompute all optical flows in " << diff.count() << " s" << std::endl;
        return flows_x.size();
    }



private:
    cv::VideoCapture capture;
    std::deque<cv::Mat> flows_x;
    std::deque<cv::Mat> flows_y;
};