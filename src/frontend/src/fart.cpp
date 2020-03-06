#include "AudioFile.h"
#include <torch/script.h>
#include "opticalflow.hpp"

#include <vector>
#include <filesystem>

torch::Tensor matToTensor(const cv::Mat& mat) {
    auto tensor = torch::from_blob(mat.data, {1, mat.rows, mat.cols, 1}, at::kByte);
    tensor = tensor.permute({{0, 3, 1, 2}});
    return tensor;
}

int main(int argc, const char* argv[]) {
    try {
        auto module = torch::jit::load(argv[1]);
        std::cout << "Loaded model! " << std::endl;
        std::vector<torch::jit::IValue> inputs(1, torch::ones({1, 6, 224, 224}));
        auto out = module.forward(inputs).toTensor();
        auto index = std::get<1>(torch::max(out, 0));
        auto i = index.item<int>();
        std::cout << "Inference: " << i << std::endl;
    }
    catch (std::exception& e) {
        std::cout << "Got error: " << e.what() << std::endl;
    }
    return 0;
}