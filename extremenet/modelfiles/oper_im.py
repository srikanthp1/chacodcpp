
import torch
import torch.utils.cpp_extension

from torch.onnx import register_custom_op_symbolic

# Define a C++ operator.
def test_custom_add():    
    # op_source = """    
    # #include <torch/script.h>    

    # torch::Tensor custom_addw(torch::Tensor self, torch::Tensor other) {
    #     return self + other;    
    # }
    # static auto registry = 
    #     torch::RegisterOperators("custom_namespace::custom_addw",&custom_addw);
    # """

        # torch::Tensor layer2 = torch::nn::functional::interpolate(layerTwo, torch::nn::functional::InterpolateFuncOptions().size(vector<int64_t>{layerOne.size(0), layerOne.size(1)}).mode(torch::kBilinear));
        # torch::Tensor layer3 = torch::nn::functional::interpolate(layerThree, torch::nn::functional::InterpolateFuncOptions().size(vector<int64_t>{layerOne.size(0), layerOne.size(1)}).mode(torch::kBilinear));
        # torch::Tensor layer4 = torch::nn::functional::interpolate(layerFour, torch::nn::functional::InterpolateFuncOptions().size(vector<int64_t>{layerOne.size(0), layerOne.size(1)}).mode(torch::kBilinear));
# layerOne.size(0), layerOne.size(1)
        # return torch::cat({layer1,layer2,layer3,layer4}, dim = 1);  
# layerTwo.size(0),layerTwo.size(1),

    op_source = """    
    #include <torch/script.h>   
    #include <vector> 

    torch::Tensor aggupscop(torch::Tensor layerOne, torch::Tensor layerTwo, torch::Tensor layerThree, torch::Tensor layerFour) {
        auto layer1 = layerOne;
        auto layer2 = torch::nn::functional::interpolate(layerTwo, torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>({layerOne.size(2), layerOne.size(3)})).mode(torch::kBilinear));
        auto layer3 = torch::nn::functional::interpolate(layerThree, torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>({layerOne.size(2), layerOne.size(3)})).mode(torch::kBilinear));
        auto layer4 = torch::nn::functional::interpolate(layerFour, torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>({layerOne.size(2), layerOne.size(3)})).mode(torch::kBilinear));

        return torch::cat({layer1,layer2,layer3,layer4}, 1);  

    }
    static auto registry = 
        torch::RegisterOperators("custom_namespace::aggupscop",&aggupscop);
    """
    torch.utils.cpp_extension.load_inline(
        name="aggupscop",
        cpp_sources=op_source,
        is_python_module=False,
        verbose=True,
    )

test_custom_add()

# Define the operator registration method and register the operator.


def symbolic_custom_add(g, layerOne, layerTwo, layerThree, layerFour):
    return g.op('custom_namespace::aggupscop', layerOne, layerTwo, layerThree, layerFour)

register_custom_op_symbolic('custom_namespace::aggupscop', symbolic_custom_add, 9)

# Build an operator model.
class CustomAddModel(torch.nn.Module):
    def forward(self, a, b,c,d):
        return torch.ops.custom_namespace.aggupscop(a, b,c ,d)

c = CustomAddModel()

# print(c(torch.randint(0,1,(2,2,416,416)).float(),torch.randint(0,1,(2,2,416,416)).float(),torch.randint(0,1,(2,2,416,416)).float(),torch.randint(0,1,(2,2,416,416)).float()))

