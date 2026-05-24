#include "core/Graph.hpp"

namespace gjwinfer{

    void  RuntimeGraph::AddAttribute(const std::string& name,const std::shared_ptr<RuntimeAttribute>& attr){
        if(runtimeattribute_.find(name) == runtimeattribute_.end()){
            runtimeattribute_[name] = attr ;
        }
    }

    bool RuntimeGraph::HasAttribute(const std::string& name)const{
        return runtimeattribute_.find(name)!=runtimeattribute_.end();
    }

    void RuntimeGraph::AddOperand(const std::string&name, std::unique_ptr<Operand> operand){
        runtimeoperand_[name] = std::move(operand);
    }

    bool RuntimeGraph::IsWeight(const std::string &name) const{
        return  runtimeattribute_.find(name)!=runtimeattribute_.end();
    }

    bool RuntimeGraph::HasOperand(const std::string & name)const{
        return runtimeoperand_.find(name)!=runtimeoperand_.end();
    }

    void RuntimeGraph::AddOperator(std::shared_ptr<Operator> ope){
        runtimeops_.push_back(ope);
    }

    std::shared_ptr<RuntimeAttribute> RuntimeGraph::GetWeight(const std::string& name){
        return runtimeattribute_[name];
    }

    Operand* RuntimeGraph::GetOperandPtr (const std::string& name){
        return runtimeoperand_[name].get();
    }
     
    void RuntimeGraph::BuildFromONNX(const std::string& modelpath){
        ONNXParser parser;
        bool success = parser.Parse(modelpath,*this);
        if(!success); //打印消息退出
        topoops_ = Toposort(runtimeops_);
        
        
    }

    std::vector<Operator*> RuntimeGraph::Toposort(const std::vector<std::shared_ptr<Operator>>& runtimeops){
        std::vector<Operator*> toposortops;
        std::map<Operator*,int> op_indegree;
        std::queue<Operator*> queops;
        //计算所有算子的入度
        for(const auto& op:runtimeops){
            int in_degree=0;
            for(const auto& operand:op->inputs){
                if(operand->producer!=nullptr) in_degree+=1;
            }
            op_indegree[op.get()] = in_degree;
            if(in_degree==0) queops.push(op.get());
        }
        //基于操作数的consumer进行排序
        while(!queops.empty()){
            Operator* current_op = queops.front();
            queops.pop();
            toposortops.push_back(current_op);
            for(const auto& operand:current_op->outputs){
                for(const auto& consumer_op:operand->consumers ){
                    op_indegree[consumer_op]-=1;
                    if(op_indegree[current_op]==0) queops.push(current_op);
                }
            }
        }
        CHECK(toposortops.size()==runtimeops.size()) <<"topo sort failed:graph has a cycle";
        return toposortops;
    }
}