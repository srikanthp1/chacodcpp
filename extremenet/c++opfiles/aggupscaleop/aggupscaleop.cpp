#include <torch/torch.h>
#include <iostream>

torch::Tensor aggupscaleop(
    torch::Tensor layerOne,
    torch::Tensor layerTwo,
    torch::Tensor layerThree,
    torch::Tensor layerFour){
        std::vector <torch::Tensor> layers(4);
        layers[0] = layerOne;
        layers[1] = torch::nn::functional::interpolate(layerTwo, torch::nn::functional::InterpolateFuncOptions().size(std::vector<int,int>{layerOne.size(0), layerOne.size(0)}).mode(torch::kBilinear));
        layers[2] = torch::nn::functional::interpolate(layerThree, torch::nn::functional::InterpolateFuncOptions().size(std::vector<int,int>{layerOne.size(0), layerOne.size(0)}).mode(torch::kBilinear));
        layers[3] = torch::nn::functional::interpolate(layerFour, torch::nn::functional::InterpolateFuncOptions().size(std::vector<int,int>{layerOne.size(0), layerOne.size(0)}).mode(torch::kBilinear));

        return torch::cat(layers, dim = 1);
    }

static auto registry = torch::RegisterOperators("mynamespace::aggupscaleop", &aggupscaleop);