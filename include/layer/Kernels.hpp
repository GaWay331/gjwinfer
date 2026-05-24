#pragma once
#include "core/RuntimeContext.hpp"

namespace gjwinfer{
class ExecutionKernel{
public:
    virtual ~ExecutionKernel() = default;

    virtual void Perpare() = 0;

    virtual void Forward(RuntimeContext& ctx) const= 0;

    
};
}