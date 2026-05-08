#pragma once 
#include "third_party/onnx/onnx.pb.h"
#include "include/core/Graph.hpp"
#include "include/core/Tensor.hpp"
namespace gjwinfer{

class ONNXParser{
public:
    bool Parse(const std::string& filse_path,RuntimeGraph &graph);
    /**
     * @brief 解析函数
     */

    void LoadInitializers(const onnx::GraphProto& graph_proto,RuntimeGraph &graph);
    /**
     * @brief 解析权重
     */
    void FindGraphInput(const onnx::GraphProto& graph_proto,RuntimeGraph &graph);
    /**
     * @brief 找出图的所有输入
     */

    void FindGraphOutput(const onnx::GraphProto& graph_proto,RuntimeGraph &graph);
    /**
     * @brief 找出图的所有输出
     */
    
    void LoadNodes(const onnx::GraphProto& graph_proto,RuntimeGraph &graph);
    /**
     * @brief 解析节点
     */

    void ParseAttributes(const onnx::GraphProto& graph_proto,RuntimeGraph &graph);
    /**
     * @brief 解析算子的参数
     */
    RuntimeParameterType MapONNXType(int onnx_type);
    /**
     * @brief 将ONNX格式下的数据枚举类型转换为gjwinfer下的数据枚举类型
     */
    DataType MapONNXDataType(int onnx_type);


};

}