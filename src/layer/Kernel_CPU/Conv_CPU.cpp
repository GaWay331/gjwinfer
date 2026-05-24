#pragma once
#include "core/Tensor.hpp"
#include <Eigen/Dense>
#include "layer/Kernel_CPU/Conv_CPU.hpp"


namespace gjwinfer{
    using MatrixXfRowMajor =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    void Conv_CPU::Forward(RuntimeContext& ctx) const {
        const Tensor& input = ctx.GetTensor(input_id_);
        Tensor& output = ctx.GetTensor(output_id_);
        ConvForwardNCHW(input,*weight_,bias_,output,params_);

    }
    void Conv_CPU::ConvForwardNCHW(const Tensor& input,const Tensor& weight,const Tensor* bias,
        Tensor& output,const ConvParams params) const{
            const auto& x_shape = input.get_shapes();
            const auto& w_shape = weight.get_shapes();
            const auto& y_shape = output.get_shapes();
            CHECK_EQ(x_shape.size(),4);
            CHECK_EQ(w_shape.size(),4);
            CHECK_EQ(y_shape.size(),4);

            uint32_t N = x_shape[0];
            uint32_t C = x_shape[1];
            uint32_t H = x_shape[2];
            uint32_t W = x_shape[3];
            uint32_t KM = w_shape[0];
            uint32_t KC = w_shape[1];
            uint32_t KH = w_shape[2];
            uint32_t KW = w_shape[3];
            uint32_t OH = y_shape[2];
            uint32_t OW = w_shape[3];

            uint32_t G = params.group;
            CHECK_EQ(C,KC*params.group)<<"kernel channels not equal input channels";
            CHECK(KM%params.group==0);
            uint32_t K = KC*KH*KW;
            uint32_t L = OH*OW;

            const float* x = input.dataptr<float>();
            const float* w = weight.dataptr<float>();
            float* y = output.dataptr<float>();
            const float* b = bias?bias->dataptr<float>():nullptr;

            std::vector<float> col_buffer(K*L);
            Eigen::Map<const MatrixXfRowMajor> weight_mat(w,KM,K);
            for(uint32_t n =0;n<N;++n){
                const float* input_ptr = n*C*H*W+x;
                float* output_ptr = y+n*KM*OH*OW;
                Im2ColNCHW(input_ptr,C,H,W,KM,KC,KH,KW,OH,OW,params,col_buffer.data());
                Eigen::Map<const MatrixXfRowMajor> col_mat(col_buffer.data(),K,L);
                Eigen::Map< MatrixXfRowMajor> output_mat(output_ptr,KM,L);
                if(params.group==1){
                    output_mat.noalias() = weight_mat *col_mat;
                    if(b){
                        for(uint32_t m = 0;m<KM;++m){
                            for(uint32_t l = 0;l<L;++l){
                                output_mat(m,l)+=b[m];
                            }
                        }
                    }
                }
                else{
                    uint32_t Cpg = C/G;
                    uint32_t KMpg = C/G;
                    for(uint32_t g = 0;g<params.group;++g){
                        uint32_t weight_row_start = g*KMpg;
                        uint32_t col_row_start = g*K;
                        uint32_t output_row_start = g*KMpg;
                        auto weight_mat_g = weight_mat.block(
                            static_cast<Eigen::Index>(weight_row_start),
                            static_cast<Eigen::Index>(0),
                            static_cast<Eigen::Index>(KMpg),
                            static_cast<Eigen::Index>(K)
                        );
                        auto col_mat_g = col_mat.block(
                            static_cast<Eigen::Index>(col_row_start),
                            static_cast<Eigen::Index>(0),
                            static_cast<Eigen::Index>(K),
                            static_cast<Eigen::Index>(L)
                        );
                        auto output_mat_g = output_mat.block(
                            static_cast<Eigen::Index>(output_row_start),
                            static_cast<Eigen::Index>(0),
                            static_cast<Eigen::Index>(KMpg),
                            static_cast<Eigen::Index>(L)
                        );
                        output_mat_g.noalias() = weight_mat_g*col_mat_g;
                        if(b){
                            for(uint32_t m = 0;m<KMpg;++m){
                                float bias_value = b[m+output_row_start];
                                for(uint32_t l = 0;l<L;++l){
                                    output_mat_g(m,l)+=bias_value;
                                }
                            }
                        }


                    }
                }
            }


    }
    void Im2ColNCHW(const float* input,uint32_t C,uint32_t H,
        uint32_t W,uint32_t KM,uint32_t KC,uint32_t KH,uint32_t KW,
        uint32_t OH,uint32_t OW ,ConvParams p, float* col){
            for(uint32_t c = 0;c<C;++c){
                for(uint32_t kh = 0;kh<KH;++kh ){
                    for(uint32_t kw = 0;kw<KW;++kw){
                        uint32_t col_r = (c*KH+kh)*KW+kw;
                        for(uint32_t oh = 0;oh<OH;++oh){
                            int ih = static_cast<int>(oh*p.stride_h+kh*p.dilation_h-p.pad_top);
                            for(uint32_t ow = 0;ow<OW;++ow){
                                int iw = static_cast<int>(ow*p.stride_w+kw*p.dilation_w-p.pad_left);
                                //Im2col中点的位置与实际一维存储后位置的映射
                                uint32_t col_idx = oh*OW+ow+col_r*OH*OW;
                                if(ih<0||ih>=static_cast<int>(H)||iw<0||iw>static_cast<int>(W)){
                                    col[col_idx] = 0.0f;
                                }
                                else{
                                    uint32_t input_idx = c*W*H+ih*W+iw;
                                    col[col_idx] = input[input_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
}
