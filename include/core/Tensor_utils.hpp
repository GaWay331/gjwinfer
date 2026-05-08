#pragma once
#include "Tensor.hpp"

namespace gjwinfer{
namespace utils{
    void TensorAdd(const Tensor &input_a,const Tensor& input_b,Tensor& output);
    /**
     * @brief 张量相加
     * @param[in] input_a张量1
     * @param[in] input_ b张量2
     * @param[out] output 输出张量
     */
    void TensorMatMUl(const Tensor &input_a,const Tensor& input_b,Tensor& output);
    /**
     * @brief 张量的每个矩阵乘法
     */
    void TensorFill(Tensor& input,float& value);
    /**
     * @brief 张量填充值
     */
    void TensorElementMUL(const Tensor &input_a,const Tensor& input_b,Tensor& output);
    /**
     * @brief 张量的对应元素相乘
     */
    void TensorFillZero(Tensor& input,float& value);

    void TensorFillOnes(Tensor& input,float& value);

}
}