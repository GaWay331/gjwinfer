#pragma once
#include "layer/ShapeInfer/ConvShapeInfer.hpp"
#include "layer/ParseParams/ParseConvParams.hpp"
#include "layer/KernelFactory.hpp"

namespace gjwinfer{
    void ConvShapeInfer::ShapeInfer(Operator& op){
        Operand* input = op.inputs[0];
        Operand* output = op.outputs[0];
        std::vector<uint32_t> input_shape = input->shapes;
        ConvParams convparmas = ParseConvParams(op);
        const auto& weight_name = op.inputs_name.at(1);
        const auto& weight = op.weights.at(weight_name);
        const std::vector<uint32_t> w_shapes = weight->weight_tensor->get_shapes();
        std::vector<uint32_t> output_shape = ConvInferShape(input_shape,w_shapes,convparmas);
        output->shapes = output_shape;//将推导的输出操作数形状赋值给operand的shape
        


    }
    std::vector<uint32_t> ConvInferShape(const std::vector<uint32_t>& x_shape,
        const std::vector<uint32_t>& w_shape,
        const ConvParams& convparams ){
            CHECK_EQ(x_shape.size(),4);
            CHECK_EQ(w_shape.size(),4);
            uint32_t C_out = w_shape[0];
            uint32_t K_C = w_shape[1];
            uint32_t K_H = w_shape[2];
            uint32_t K_W = w_shape[3];

            uint32_t N = x_shape[0];
            uint32_t C_IN = x_shape[1];
            uint32_t I_H = x_shape[2];
            uint32_t I_W = x_shape[3];
            CHECK_EQ(C_IN,convparams.group*K_C);
            uint32_t h = (convparams.pad_bottom+convparams.pad_top+I_H-convparams.dilation_h*(K_H-1)-1)/
                            convparams.stride_h+1;
            uint32_t w = (convparams.pad_right+convparams.pad_left+I_H-convparams.dilation_w*(K_W-1)-1)/
                            convparams.stride_w+1;
            return std::vector<uint32_t>{N,C_out,h,w};
        }
    REGISTER_SHAPE_KERNEL("Conv",ConvShapeInfer);

}