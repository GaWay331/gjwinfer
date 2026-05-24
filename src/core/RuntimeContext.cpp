#include "core/RuntimeContext.hpp"

namespace gjwinfer{

    Tensor& RuntimeContext::GetTensor(int id){
        return tensors_[id];
    }

    const Tensor& RuntimeContext::GetTensor(int id) const{
        return tensors_[id];
    }
}