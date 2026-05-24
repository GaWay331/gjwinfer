#pragma once
#include "Tensor.hpp"



namespace gjwinfer{

class RuntimeContext{

public:
    explicit RuntimeContext(std::vector<Tensor> tensors)
        :tensors_(std::move(tensors)){}

    Tensor& GetTensor(int id);
    
    const Tensor& GetTensor(int id) const;
private:
    std::vector<Tensor> tensors_;//操作数
};
}