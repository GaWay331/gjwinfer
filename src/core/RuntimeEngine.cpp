#include "core/RuntimeEngine.hpp"


namespace gjwinfer{
    RuntimeContext RuntimeEngine::CreatContext(const std::vector<TensorDesc>& tensor_descs) const{
        std::vector<Tensor> tensors;
        tensors.reserve(tensor_descs.size());
        for(const TensorDesc& tensor_desc:tensor_descs){
            tensors.emplace_back(tensor_desc.shape,tensor_desc.data_type,tensor_desc.device_type);
        }
        return RuntimeContext(std::move(tensors));

    }

    void RuntimeEngine::Forward(RuntimeContext& ctx) const{
        for(auto& kernel:runtimekernels_){
            kernel->Forward(ctx);
        }
    }

    int RuntimeEngine::AddTensorDesc(const TensorDesc& tensordesc){
        int id = static_cast<int>(tensor_descs_.size());
        tensor_name_to_id_[tensordesc.name] = id;
        tensor_descs_.push_back(tensordesc);
        return id;
    }

    int RuntimeEngine::GetTensorId(const std::string& name) const{
        return tensor_name_to_id_.at(name);
    }

    void RuntimeEngine::AddKernel(const std::unique_ptr<ExecutionKernel> kernel){
            runtimekernels_.push_back(std::move(kernel));
    }
}