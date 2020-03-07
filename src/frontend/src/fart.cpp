#include "AudioFile.h"
#include <torch/torch.h>
#include <torch/script.h>
#include "opticalflow.hpp"

#include <chrono>
#include <vector>
#include <filesystem>

constexpr size_t frames = 3;

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: Fart.exe [model path] [video path]" << std::endl;
    }

    try {
        torch::jit::script::Module module = torch::jit::load(argv[1]);
        auto cuda = torch::cuda::is_available();
        if (cuda) {
            module.to(at::kCUDA);
        }
        std::cout << "Loaded model! " << std::endl;

        std::string video(argv[2]);
        opticalflow loader(video);
        loader.preComputeAll();

        std::cout << "Collecting all inference..." << std::endl;
        auto start = std::chrono::steady_clock::now();
        std::vector<int> outputs;
        size_t count = 0;
        while (auto t = loader.next(frames)) {
            count++;
            auto input_tensor = (*t).sub(0.5).div(0.5);
            if (cuda) {
                input_tensor = input_tensor.to(at::kCUDA);
            }
            std::vector<torch::jit::IValue> input;
            input.emplace_back(std::move(input_tensor));
            auto out = module.forward(input).toTensor().cpu().view(2);
            auto i = std::get<1>(torch::max(out, 0));
            auto index = i.item<int>();
            outputs.emplace_back(std::move(index));
        }
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "Collected " << count << " inference in " << diff.count() << " s" << std::endl;
        std::cout << "Inference are: " << std::endl;
        for (auto i : outputs) {
            std::cout << i;
        }
        std::cout << std::endl;
    }
    catch (std::exception& e) {
        std::cout << "Got error: " << e.what() << std::endl;
    }
    return 0;
}