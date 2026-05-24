#pragma once
#include <vector>
#include <numeric>
#include "Tensor.hpp"
#include "Parameter.hpp"
namespace gjwinfer{

class RuntimeAttribute{
public:
    RuntimeAttribute(const std::string& name):name_{name}{};

    std::string name_;

    std::vector<char> weight_data;

    std::shared_ptr<Tensor> weight_tensor=nullptr;



};

}