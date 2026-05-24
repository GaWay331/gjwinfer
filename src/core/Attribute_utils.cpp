#include "core/Attribute_utils.hpp"

namespace gjwinfer{
    bool HasAttribute(const Operator& op ,const std::string& name){
        return op.attributes.find(name)!=op.attributes.end();
    }

    int GetInt(const Operator& op,
        const std::string& name,
        int default_value){
            auto iter = op.attributes.find(name);
            if(iter == op.attributes.end()) return default_value;
            CHECK(iter->second!=nullptr);
            CHECK(iter->second->type == RuntimeParameterType::kParameterInt);
            auto param = std::dynamic_pointer_cast<RuntimeParameterInt>(iter->second);
            int value = param->value;
            return value;
        }
    
    float GetFloat(const Operator& op,
            const std::string& name,
            float default_value){
            auto iter = op.attributes.find(name);
            if(iter == op.attributes.end()) return default_value;
            CHECK(iter->second!=nullptr);
            CHECK(iter->second->type == RuntimeParameterType::kParameterFloat);
            auto param = std::dynamic_pointer_cast<RuntimeParameterFloat>(iter->second);
            float value = param->value;
            return value;
            }
    
    std::string GetString(const Operator& op,const std::string& name,const std::string& default_value){
        auto iter = op.attributes.find(name);
        if(iter == op.attributes.end()) return default_value;
        CHECK(iter->second!=nullptr);
        CHECK(iter->second->type == RuntimeParameterType::kParameterString);
        auto param = std::dynamic_pointer_cast<RuntimeParameterString>(iter->second);
        std::string value = param->value;
        return value; 
    }

    
    std::vector<int> GetIntArray(const Operator& op,
                            const std::string& name, 
                            const std::vector<int>& default_value){
                                auto iter = op.attributes.find(name);
                                if(iter ==op.attributes.end()) return default_value;
                                CHECK(iter->second!=nullptr);
                                CHECK(iter->second->type == RuntimeParameterType::kParameterIntArray);
                                auto param = std::dynamic_pointer_cast<RuntimeParameterIntArray>(iter->second);
                                std::vector<int> value = param->value;
                                return value; 
    }
                         
                    
                        
    
    std::vector<float> GetFloatArray(const Operator& op,
                              const std::string& name,
                              const std::vector<float>& default_value){
                                auto iter = op.attributes.find(name);
                                if(iter ==op.attributes.end()) return default_value;
                                CHECK(iter->second!=nullptr);
                                CHECK(iter->second->type == RuntimeParameterType::kParameterFloatArray);
                                auto param = std::dynamic_pointer_cast<RuntimeParameterFloatArray>(iter->second);
                                std::vector<float> value = param->value;
                                return value; 
                              }
}