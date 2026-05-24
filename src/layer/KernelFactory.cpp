#include "layer/KernelFactory.hpp"

namespace gjwinfer{
    void KernelFactory::Register(const std::string& name,DeviceType device,KernelCreator kernelcreator){
        RegistryKey key = std::make_tuple(name,device);
        kernelregistry_[key] = kernelcreator;
    }

    std::unique_ptr<ExecutionKernel> KernelFactory::CreateKernel(const std::string& name,DeviceType device){
        RegistryKey key = std::make_tuple(name,device);
        if(kernelregistry_.find(key)==kernelregistry_.end()){
            if(device==DeviceType::Device_GPU){
                return CreateKernel(name,DeviceType::Device_CPU);
            }
            return nullptr;
        }
        return kernelregistry_[key]();
    }
   void ShapeInferFactory::Register(const std::string& name,ShapeInferCreator s_creator){
        shapeifregistry_[name] = s_creator;
   }

   std::unique_ptr<ShapeInferKernel> ShapeInferFactory::CreateShapeKernel(const std::string& op_name){
    CHECK(shapeifregistry_[op_name]!=nullptr);
    return shapeifregistry_[op_name]();
   }
}