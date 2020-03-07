#include "AudioFile.h"
#include <torch/script.h>
#include "opticalflow.hpp"

#include <vector>
#include <filesystem>

int main(int argc, const char* argv[]) {
    try {
        std::string video(argv[2]);
        opticalflow loader(video);
        loader.preComputeAll();
        auto module = torch::jit::load(argv[1]);
        std::cout << "Loaded model! " << std::endl;
        std::vector<torch::jit::IValue> inputs(1, torch::ones({1, 6, 224, 224}));
        auto out = module.forward(inputs).toTensor().view(2).sub(0.5).div(0.5);
        auto index = std::get<1>(torch::max(out, 0));
        auto i = index.item<int>();
        std::cout << "Inference: " << i << std::endl;
    }
    catch (std::exception& e) {
        std::cout << "Got error: " << e.what() << std::endl;
    }
    return 0;
}