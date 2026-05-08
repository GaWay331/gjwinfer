#include <glog/logging.h>
#include <gtest/gtest.h>
#include "core/Tensor.hpp"
#include "core/Tensor_utils.hpp"

using namespace gjwinfer;

TEST(TensorUtilsTest, TensorAddTest){
    std::vector<uint32_t> shapes1={1,2,4,5};
    std::vector<uint32_t> shapes2={1,2,4,5};
    Tensor a(shapes1);
    Tensor b(shapes2);
    Tensor c(shapes2);
    float m=5;
    float n=10;
    utils::TensorFill(a,m);
    utils::TensorFill(b,n);
    utils::TensorAdd(a,b,c);
    float* ptr_out = c.size() > 0 ? c.raw_ptr() : nullptr;
    ASSERT_NE(ptr_out,nullptr);
    for(int i=0;i<c.size();i++){
        EXPECT_FLOAT_EQ(ptr_out[i],15.0f);
    }

}
