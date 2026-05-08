# pragma once
# include "Tensor.hpp"
#include <string>
#include "Operator.hpp"


namespace gjwinfer{
class Operand{
public:
    std::string name;

    std::vector<uint32_t> shapes;

    std::shared_ptr<Tensor> tensor_ptr =nullptr;

    RuntimeParameterType data_type = RuntimeParameterType::kParameterFloat;

    Operator * producer =nullptr;

    std::vector<Operator*> consumers;//会使用该操作数的OPERATOR

    int consumer_ref_count = 0;//用于内存池判断是否释放
    
    
    
};

}