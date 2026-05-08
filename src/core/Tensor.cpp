#include "core/Tensor.hpp"

namespace gjwinfer{
    static size_t GetElementSize(DataType datatype){
        switch(datatype){
            case(DataType::kFloat32) : return 4;
            default: LOG(FATAL)<<"unknown type";  return 0;
        }
    }
    size_t Tensor::byte_size()const{
        uint32_t size_element = this->size();
        return size_element*GetElementSize(data_type_);
    }

    Tensor::Tensor(const std::vector<uint32_t> &shapes,DataType datatype,DeviceType device_type)
    :shapes_{shapes},data_type_{datatype}, device_type_{device_type}{
        size_t byte_size = this->byte_size();
        if(byte_size==0) return;
        if(device_type_==DeviceType::Device_CPU){
            void* ptr =std::malloc(byte_size);
            CHECK_NOTNULL(ptr);
            data_=std::shared_ptr<void>(ptr,[](void* p){
                std::free(p);
            });
        }
    }
    Tensor::Tensor(const std::vector<uint32_t>& shapes,void* external_ptr,DataType datatype,DeviceType device_type)
    :shapes_{shapes},data_type_{datatype},device_type_{device_type}{
        if(external_ptr!=nullptr){
            data_ = std::shared_ptr<void>(external_ptr,[](void* p){});
        }
    }

    uint32_t Tensor::size() const{
        if(shapes_.empty()) return 0;
        return std::accumulate(shapes_.begin(),shapes_.end(),1,std::multiplies<uint32_t>());
    }
    uint32_t Tensor::size_at(uint32_t index) const{
        if(shapes_.empty()) return 0;
        return shapes_.at(index);
    }

    void* Tensor::raw_ptr(){
        return data_.get();
    }

    const void* Tensor::raw_ptr() const{
        if(data_==nullptr) return nullptr;
        return data_.get();
    }

    bool Tensor::empty() const{
        return data_==nullptr;
    }

    void Tensor::reshape(const std::vector<uint32_t> &new_shapes){
        uint32_t current_size=this->size();
        uint32_t new_size = std::accumulate(new_shapes.begin(),
        new_shapes.end(),1,std::multiplies<uint32_t>());
        CHECK_EQ(current_size,new_size)<<"RESHAPE FAILED:size missmatch!";
        this->shapes_=new_shapes;
    }
    const std::vector<uint32_t>& Tensor::get_shapes() const{
        return this->shapes_;
    }

}