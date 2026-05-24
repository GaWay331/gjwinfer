//用于解析算子的相关配置
#pragma once 
#include "Operator.hpp"

namespace gjwinfer{
bool HasAttribute(const Operator& op ,const std::string& name);//判断是否有对应的参数

int GetInt(const Operator& op,
    const std::string& name,
    int default_value);

float GetFloat(const Operator& op,
        const std::string& name,
        float default_value);

std::string GetString(const Operator& op,
               const std::string& name,
               const std::string& default_value);

std::vector<int> GetIntArray(const Operator& op,
                      const std::string& name,
                      const std::vector<int>& default_value);

std::vector<float> GetFloatArray(const Operator& op,
                          const std::string& name,
                          const std::vector<float>& default_value);


}
