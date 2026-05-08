#pragma once 
#include "Operator.hpp"
#include "Attribute.hpp"
#include "parser/ONNXParser.h"
#include "queue"

namespace gjwinfer{
class  RuntimeGraph{
private:
    std::vector<std::shared_ptr<Operator>> runtimeops_;//所有的算子

    std::map<std::string,std::unique_ptr<Operand>> runtimeoperand_;//模型所有涉及的操作数

    std::map<std::string,std::shared_ptr<RuntimeAttribute>> runtimeattribute_;//所有的权重

    std::map<std::string,std::unique_ptr<Operand>> inputoperands_;//模型的输入操作数

    std::map<std::string,std::unique_ptr<Operand>> outputoperands_;//模型的输出

    std::string modelpath_;

    std::vector<Operator*> topoops_;

    //std::unique_ptr<MemoryPool> allocator_;
public:

    //friend class ONNXParser;

    std::vector<Operator*> Toposort(const std::vector<std::shared_ptr<Operator>>& runtimeops);
    /**
     * @brief 用于算子排序
     * 
     */
    void Forward();
    /**
     * @brief 模型推理
     */
    void BuildFromONNX(const std::string& modelpath);
    /**
     * @构建自己的graph
     */

    void AddAttribute(const std::string& name,const std::shared_ptr<RuntimeAttribute> & attr);
    
    bool HasAttribute(const std::string & name) const;

    void AddOperand(const std::string&name, std::unique_ptr<Operand> operand);

    void AddOperator(std::shared_ptr<Operator> ope);

    bool IsWeight(const std::string &name) const;
    /**
     * @brief 用于确认onnx的node输入是否为权重
     */

    bool HasOperand (const std::string& name)const;

    std::shared_ptr<RuntimeAttribute> GetWeight(const std::string& name);
    /**
     * @brief 获取权重的智能指针
     */

    Operand* GetOperandPtr(const std::string& name);
};

}

