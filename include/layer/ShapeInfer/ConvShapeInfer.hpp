#pragma once
#include "ShapeInfer.hpp"
#include "ParseParams/ParseConvParams.hpp"

namespace gjwinfer{

class ConvShapeInfer:public ShapeInferKernel{
public:
    virtual ~ConvShapeInfer() = default;

    virtual  void ShapeInfer(Operator& op) override;
};

std::vector<uint32_t> ConvInferShape(const std::vector<uint32_t>& x_shape,
                                    const std::vector<uint32_t>& w_shape,
                                    const ConvParams& convparams );
}