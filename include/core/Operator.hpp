#pragma once
#include "Operand.hpp"
#include <map>
#include "Attribute.hpp"
#include "layer/Kernels.hpp"

namespace gjwinfer{
class Operator{
public:

    std::string name;

    std::string op_type;

    std::map<std::string,std::shared_ptr<RuntimeParameter>> attributes;

    std::map<std::string,std::shared_ptr<RuntimeAttribute>> weights;
    

    std::map<std::string,Operand*> inputs_map;

    std::map<std::string,Operand*> outputs_map;    

    std::vector<Operand*> inputs;

    std::vector<Operand*> outputs;

    std::unique_ptr<ExecutionKernel> kernel=nullptr;


};
}