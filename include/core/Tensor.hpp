#pragma once
#include <memory>
#include <vector>
#include <cstdint>
#include <numeric>
#include <glog/logging.h>

namespace gjwinfer{
enum class DeviceType{
    Device_CPU =0,
    Device_GPU =1
};
enum class DataType{
    kUnkonwn = 0,
    kFloat32 = 1
};

class Tensor{
public:
    Tensor()=default;
    explicit Tensor(const std::vector<uint32_t>& shapes,DataType datatype=DataType::kFloat32,DeviceType device_type = 
        DeviceType::Device_CPU);
    /**
     * @brief 构造函数，基于传入的形状分配内存
     * @param shpes为数据的形状
     */

    explicit Tensor(const std::vector<uint32_t>& shapes,void* external_ptr,DataType datatype=DataType::kFloat32,DeviceType device_type =
         DeviceType::Device_CPU);
    ~Tensor()=default;
    
    template <typename T>
    T* dataptr() {
        return static_cast<T*> (data_.get());
    }

    template <typename T>
    const T* dataptr() const{
        return static_cast<const T*> (data_.get());
    }

    void* raw_ptr();
    const void* raw_ptr() const;
    /**
     * @brief 获取tensor指向的指针
     * 
     */

    uint32_t size() const;
    /**
     * @brief 获取tensor元素的个数
     */

    size_t byte_size() const;
    /**
     * @brief 计算字节数
     */
    uint32_t size_at(uint32_t index) const;
    /**
     * @brief 获取在第几个维度的大小
     */

    bool empty() const;
    /**
     * @brief 判断tensor是否为空
     */
    void reshape(const std::vector<uint32_t>& new_shapes);
    /**
     * @brief 重新修改tensor
     */
    const std::vector<uint32_t>& get_shapes() const;
    /**
     * @brief 获取张张量的形状
     */
private:
    std::vector<uint32_t> shapes_;
    std::shared_ptr<void> data_;
    DeviceType device_type_ = DeviceType::Device_CPU;
    DataType data_type_ =DataType::kFloat32;
};

}