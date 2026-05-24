#pragma once
#include "Kernels.hpp"
#include "KernelFactory.hpp"
#include "ParseParams/ParseConvParams.hpp"

namespace gjwinfer{
class Conv_CPU : public ExecutionKernel{
public:
    Conv_CPU(int input_id,int output_id,const Tensor* weight,const Tensor* bias,ConvParams params )
    :input_id_(input_id),output_id_(output_id),weight_(weight),bias_(bias),params_(params){}

    void Forward(RuntimeContext& ctx) const override;

    void ConvForwardNCHW(const Tensor& input,const Tensor& weight,const Tensor* bias,
        Tensor& output,const ConvParams params) const ;
private:
    int input_id_ = -1;
    int output_id_ = -1;
    const Tensor* weight_ = nullptr;
    const Tensor* bias_ =nullptr;
    
    ConvParams params_;
};
void Im2ColNCHW(const float* input,uint32_t C,uint32_t H,
    uint32_t W,uint32_t KM,uint32_t KC,uint32_t KH,uint32_t KW,
    uint32_t OH,uint32_t OW ,ConvParams p, float* col);//对一个batch的输入实现im2col
    /**
     * @brief 将卷积操作的特征图转化为列的矩阵
     * @param input 特征图的指针
     * @param C、H、W、 特征图的通道、高度、宽度
     * @param KM KC KH KW 卷积核的个数，通道，高度，宽度
     * @param OH OW 卷积操作后的输出特征图的高度、宽度
     * @param p卷积算子的参数
     * @param col Im2col处理后结果存放的位置
     */
    
    
}