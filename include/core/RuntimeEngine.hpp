#pragma once
#include "Tensor.hpp"
#include <unordered_map>
#include "layer/Kernels.hpp"
#include "RuntimeContext.hpp"


namespace gjwinfer{

class RuntimeEngine{
public:
    RuntimeContext CreatContext(const std::vector<TensorDesc>& tensor_descs) const;

    void Forward(RuntimeContext& ctx) const;

    int AddTensorDesc(const TensorDesc& tensordesc);

    int GetTensorId(const std::string& name) const;

    void AddKernel(const std::unique_ptr<ExecutionKernel> kernel);


private:
    std::vector<TensorDesc> tensor_descs_;
    std::unordered_map<std::string,int> tensor_name_to_id_;
    std::vector<std::unique_ptr<ExecutionKernel>> runtimekernels_;
};
}