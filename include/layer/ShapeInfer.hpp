#pragma once
#include "core/Operator.hpp"
namespace gjwinfer{

class ShapeInferKernel{
public:
    virtual ~ShapeInferKernel() = default;

    virtual  void ShapeInfer(Operator& op) = 0;
};
}
