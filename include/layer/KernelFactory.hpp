#pragma once
#include <map>
#include <tuple>
#include <string>
#include <memory>
namespace gjwinfer{

class KernelFactory{
private: 
    KernelFactory() = default;

public:
    static KernelFactory& GetInstance(){
        static KernelFactory instance;
        return instance;
    }

};
}