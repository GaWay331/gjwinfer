#pragma once
#include <map>
#include <tuple>
#include <string>
#include <memory>
#include <functional>
#include "Kernels.hpp"
#include "core/Tensor.hpp"
#include "ShapeInfer.hpp"

namespace gjwinfer{
using KernelCreator = std::function<std::unique_ptr<ExecutionKernel>()>;
using RegistryKey = std::tuple<std::string,DeviceType>;
using KernelRegistry = std::map<RegistryKey,KernelCreator>;
using ShapeInferCreator = std::function<std::unique_ptr<ShapeInferKernel>()>;//使用func包装一个类
using ShapeInferRegistry = std::map<std::string,ShapeInferCreator>;

class KernelFactory{

private: 
    KernelFactory() = default;
    KernelRegistry kernelregistry_;

public:
    static KernelFactory& GetInstance(){
        static KernelFactory instance;
        return instance;
    }
    void Register(const std::string& name,DeviceType device,KernelCreator kernelcreator);

    std::unique_ptr<ExecutionKernel> CreateKernel(const std::string& name,DeviceType device);
};

class KernelRegister{
public:
    KernelRegister(const std::string& op_name,DeviceType device,KernelCreator creator){
        KernelFactory::GetInstance().Register(op_name,device,creator);
    }

} ;

#define REGISTER_KERNEL_CPU(op_type,class_name) \
    static KernelRegister g_register_##class_name##_cpu(op_type,DeviceType::Device_CPU,[](){ \
        return std::make_unique<class_name>();\
    });

#define REGISTER_KERNEL_GPU(op_type,class_name) \
    static KernelRegister g_register_##class_name##_gpu(op_type,DeviceType::Device_GPU,[](){ \
        return std::make_unique<class_name>();\
    });

class ShapeInferFactory{
private:
    ShapeInferFactory() = default;
    ShapeInferRegistry shapeifregistry_;
public:
    static ShapeInferFactory& GetInstance(){
        static ShapeInferFactory instance;
        return instance;
    }
    void Register(const std::string& name,ShapeInferCreator creator);

    std::unique_ptr<ShapeInferKernel> CreateShapeKernel(const std::string& op_type);
};
class ShapeRegister{
public:    
    ShapeRegister(const std::string& name,ShapeInferCreator s_creator){
        ShapeInferFactory::GetInstance().Register(name,s_creator);
    }
};
#define REGISTER_SHAPE_KERNEL(op_type,class_name) \
    static ShapeRegister s_register_##class_name(op_type,[](){\
        return std::make_unique<class_name>();\
    })
}