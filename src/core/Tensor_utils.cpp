#include "core/Tensor_utils.hpp"
#include <Eigen/Dense>
namespace gjwinfer{
namespace utils{
    using MatrixXfr=Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;

    void TensorAdd(const Tensor& input_a,const Tensor& input_b,Tensor& output){
        CHECK_EQ(input_a.size(),input_b.size());
        CHECK_EQ(input_a.size(),output.size());
        CHECK(input_a.get_shapes()==input_b.get_shapes());
        CHECK(input_a.get_shapes()==output.get_shapes());
        Eigen::Map<const Eigen::ArrayXf> map_a(input_a.dataptr<float>(),input_a.size());
        Eigen::Map<const Eigen::ArrayXf> map_b(input_b.dataptr<float>(),input_b.size());
        Eigen::Map<Eigen::ArrayXf> map_out(output.dataptr<float>(),output.size());
        map_out = map_a + map_b;
    }

    void TensorMatMUl(const Tensor &input_a,const Tensor& input_b,Tensor& output){
        CHECK_EQ(input_a.get_shapes().size(), 4)<<"only 4D tensors are suportted now";
        CHECK_EQ(input_b.get_shapes().size(), 4);
        CHECK_EQ(output.get_shapes().size(), 4);
        const uint32_t batch = output.size_at(0);
        const uint32_t channels =output.size_at(1);
        const uint32_t inputa_r = input_a.size_at(2);
        const uint32_t inputa_c = input_a.size_at(3);
        const uint32_t inputb_c =input_b.size_at(3);
        //检查矩阵运算的维度是否匹配
        CHECK_EQ(inputa_r,output.size_at(2));
        CHECK_EQ(inputa_c,input_b.size_at(2));
        CHECK_EQ(inputb_c,output.size_at(3));
        //检查batch,channel是否匹配
        CHECK_EQ(batch,input_a.size_at(0));
        CHECK_EQ(batch,input_b.size_at(0));
        CHECK_EQ(channels,input_a.size_at(1));
        CHECK_EQ(channels,input_b.size_at(1));
        //计算偏移量
        const uint32_t stride_a = inputa_r*inputa_c;
        const uint32_t stride_b = inputa_c*inputb_c;
        const uint32_t stride_out = inputa_r*inputb_c;
        //进行矩阵运算
        for(uint32_t b=0;b<output.size_at(0);++b){
            for(uint32_t c=0;c<output.size_at(1);++c){
                uint32_t offset = b*channels+c;
                Eigen::Map<const MatrixXfr> map_a(input_a.dataptr<float>()+offset*stride_a,inputa_r,inputa_c);
                Eigen::Map<const MatrixXfr> map_b(input_b.dataptr<float>()+offset*stride_b,inputa_c,inputb_c);
                Eigen::Map<MatrixXfr> mapout(output.dataptr<float>()+offset*stride_out,inputa_r,inputb_c);
                mapout.noalias()=map_a*map_b;
            }
        }
    }

    void TensorFill (Tensor&input,float& value){
        Eigen::Map<Eigen::ArrayXf>map_input (input.dataptr<float>(),input.size());
        map_input.setConstant(value);
    }
    void TensorElementMUL (const Tensor &input_a,const Tensor& input_b,Tensor& output){
        CHECK_EQ(input_a.size(),input_b.size())<<"input tensors donts match";
        CHECK_EQ(input_a.size(),output.size())<<"output tensors donts match";
        CHECK(input_a.get_shapes()==input_b.get_shapes())<<"input tensors donts match";
        CHECK(output.get_shapes()==input_b.get_shapes())<<"output tensors donts match";
        Eigen::Map<const Eigen::ArrayXf> map_inputa (input_a.dataptr<float>(),input_a.size());
        Eigen::Map<const Eigen::ArrayXf> map_inputb (input_b.dataptr<float>(),input_b.size());
        Eigen::Map< Eigen::ArrayXf> map_output (output.dataptr<float>(),output.size());
        map_output=map_inputa+map_inputb;
    }
    void TensorFillZero(Tensor& input,float& value){
        Eigen::Map< Eigen::ArrayXf> map (input.dataptr<float>(),input.size());
        map.setZero();
    }

    void TensorFillOnes(Tensor& input,float& value){
        Eigen::Map< Eigen::ArrayXf> map (input.dataptr<float>(),input.size());
        map.setOnes();
    }
}  
}