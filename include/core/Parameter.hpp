#pragma once
#include <string>
#include <vector>

namespace gjwinfer{
enum class RuntimeParameterType{
    kParameterUnknown=0,
    kParameterBool = 1,
    kParameterInt = 2,
    kParameterFloat = 3,
    kParameterString = 4,
    kParameterIntArray = 5,
    kParameterFloatArray = 6
};
class RuntimeParameter{
public:
    explicit RuntimeParameter(RuntimeParameterType type = RuntimeParameterType::kParameterUnknown)
        :type(type){}

    virtual ~RuntimeParameter() = default;

    RuntimeParameterType type;

} ;
class RuntimeParameterInt:public RuntimeParameter{
public:
    explicit RuntimeParameterInt(int  param_value = 0)
        :RuntimeParameter(RuntimeParameterType::kParameterInt),value(param_value){}
    int value;
};

class RuntimeParameterFloat : public RuntimeParameter {
    public:
        explicit RuntimeParameterFloat(float param_value = 0.0f)
            : RuntimeParameter(RuntimeParameterType::kParameterFloat), value(param_value) {}
        float value;
    };
    
    // 存整数数组 (比如 stride=[1, 1], padding=[1, 1, 1, 1])
    class RuntimeParameterIntArray : public RuntimeParameter {
    public:
        // 使用 std::move 完美转发，零拷贝！
        explicit RuntimeParameterIntArray(std::vector<int> param_value)
            : RuntimeParameter(RuntimeParameterType::kParameterIntArray), 
              value(std::move(param_value)) {}
        std::vector<int> value;
    };
    class RuntimeParameterFloatArray : public RuntimeParameter {
        public:
            // 使用 std::move 完美转发，零拷贝！
            explicit RuntimeParameterFloatArray(std::vector<float> param_value)
                : RuntimeParameter(RuntimeParameterType::kParameterFloatArray), 
                  value(std::move(param_value)) {}
            std::vector<float> value;
        };
    // 存字符串 (比如 padding_mode="zeros")
    class RuntimeParameterString : public RuntimeParameter {
    public:
        explicit RuntimeParameterString(std::string param_value)
            : RuntimeParameter(RuntimeParameterType::kParameterString), 
              value(std::move(param_value)) {}
        std::string value;
    };

}