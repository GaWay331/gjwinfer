#include "parser/ONNXParser.h"
#include <fstream>

namespace gjwinfer{
    RuntimeParameterType ONNXParser::MapONNXType(int onnx_type){
        switch (onnx_type)
        {
        case onnx::TensorProto_DataType_FLOAT: return RuntimeParameterType::kParameterFloat;
        case onnx::TensorProto_DataType_INT32: return RuntimeParameterType::kParameterInt;
        default:
            return RuntimeParameterType::kParameterUnknown;
        }
    }
    DataType ONNXParser::MapONNXDataType(int onnx_type){
        switch (onnx_type)
        {
        case onnx::TensorProto_DataType_FLOAT: return DataType::kFloat32;

        default:
            return DataType::kUnkonwn;
        }
    }

    bool ONNXParser::Parse(const std::string& filse_path,RuntimeGraph &graph){
        //open file
        std::ifstream is(filse_path,std::ios::in|std::ios::binary);
        if(!is.is_open()) return false;

        onnx::ModelProto model;
        if(!model.ParseFromIstream(&is)) return false;

        const onnx::GraphProto& graph_proto = model.graph();
        LoadInitializers(graph_proto,graph);
        FindGraphInput(graph_proto,graph);
        LoadNodes(graph_proto,graph);
        FindGraphOutput(graph_proto,graph);
        return true;

    }


    void ONNXParser::LoadInitializers(const onnx::GraphProto& graph_proto,RuntimeGraph &graph){
        for(const auto& tensor_proto:graph_proto.initializer()){
            std::string name = tensor_proto.name();
            std::vector<uint32_t> shapes;
            for(int i =0;i<tensor_proto.dims_size();++i){
                int64_t dim = tensor_proto.dims(i);
                CHECK(dim>=0&&dim<=std::numeric_limits<uint32_t>::max())
                    <<"onnx dimension"<<dim<<"is out of range for uint32_t";
                shapes.push_back(static_cast<uint32_t>(dim));
            }
            int type_val = tensor_proto.data_type();
            DataType datatype = MapONNXDataType(type_val);
            auto tensor = std::make_shared<Tensor>(shapes,datatype);
            if(tensor_proto.has_raw_data()){
                const std::string& bytes = tensor_proto.raw_data();
                size_t byte_size = bytes.size();
                size_t tensor_size = tensor.get()->byte_size();
                CHECK_EQ(byte_size,tensor->byte_size());
                std::memcpy(tensor->raw_ptr(),bytes.data(),byte_size);
            }
            else{
                //处理数据不是rawdata时的情况
                if(type_val==onnx::TensorProto_DataType_FLOAT){
                    size_t count = tensor_proto.float_data_size();
                    CHECK_EQ(count,tensor->size());
                    float* tensor_ptr = tensor->dataptr<float>();
                    for (size_t i = 0; i < count; ++i) {
                        tensor_ptr[i] = tensor_proto.float_data(i);
                    }
                }
            }
            std::shared_ptr<RuntimeAttribute> attribute =std::make_shared<RuntimeAttribute>(name);
            attribute->weight_tensor = tensor;
            graph.AddAttribute(name,attribute);
        }
    }
    void ONNXParser::FindGraphInput(const onnx::GraphProto& graph_proto,RuntimeGraph &graph){
        for(const auto& input_proto :graph_proto.input()){
            std::string name = input_proto.name();
            if(graph.HasAttribute(name)) continue;

            const auto & type_proto = input_proto.type();
            CHECK(type_proto.has_tensor_type())<<"input must be a tensor type";
            const auto & tensor_proto = type_proto.tensor_type();

            int type = tensor_proto.elem_type();

            std::vector<uint32_t> shapes;
            if(tensor_proto.has_shape()){
                const auto& shape_proto = tensor_proto.shape();
                for(int i =0;i<shape_proto.dim_size();++i){
                    const auto & dim = shape_proto.dim(i);
                    if(dim.has_dim_value()){
                        int64_t dim_val = dim.dim_value();
                        CHECK(dim_val>=0)<<"dim value can not be negative";
                        shapes.push_back(static_cast<uint32_t>(dim_val));
                    } 
                    else{
                        shapes.push_back(1);
                    }
                }

            }
            auto input_operand = std::unique_ptr<Operand>();
            input_operand->name = name;
            input_operand->shapes = shapes;
            input_operand->data_type = MapONNXType(type);
            graph.AddOperand(name,std::move(input_operand));

        }
    }

    void ONNXParser::LoadNodes(const onnx::GraphProto& graph_proto,RuntimeGraph &graph){
        for(const auto& node_proto : graph_proto.node()){
            auto runtime_op = std::make_shared<Operator>() ;
            runtime_op->op_type = node_proto.op_type();
            runtime_op->name = node_proto.name();
            graph.AddOperator(runtime_op);
            //ONNX输入的转换
            for(const auto& input_name : node_proto.input()){
                if(graph.IsWeight(input_name)){
                    runtime_op->weights[input_name] = graph.GetWeight(input_name);
                    continue;
                }

                if(!graph.HasOperand(input_name)){
                    auto runtime_operand = std::make_unique<Operand>();
                    runtime_operand->name = input_name;
                    runtime_operand->consumer_ref_count+=1;
                    runtime_operand->consumers.push_back(runtime_op.get());
                    runtime_op->inputs.push_back(runtime_operand.get());
                    graph.AddOperand(input_name,std::move(runtime_operand));
                }
                else{
                    Operand* runtime_operand = graph.GetOperandPtr(input_name);
                    runtime_operand->consumer_ref_count+=1;
                    runtime_operand->consumers.push_back(runtime_op.get());
                    runtime_op->inputs.push_back(runtime_operand);
                }
            }
            //ONNX格式输出的转换
            for(const auto& output_name:node_proto.output()){
                if(!graph.HasOperand(output_name)){
                    auto runtime_operand = std::make_unique<Operand>();
                    runtime_operand->name = output_name;
                    runtime_operand->producer=runtime_op.get();
                    runtime_op->outputs.push_back(runtime_operand.get());
                    graph.AddOperand(output_name,std::move(runtime_operand));
                }
                else{
                    Operand* runtime_operand = graph.GetOperandPtr(output_name);
                    runtime_operand->producer=runtime_op.get();
                    runtime_op->outputs.push_back(runtime_operand);
                }
            }
            //ONNX节点的配置
            for(const auto& attr_proto:node_proto.attribute()){
                const std::string& name = attr_proto.name();

                switch(attr_proto.type()){
                    case onnx::AttributeProto_AttributeType_INT:{
                        runtime_op->attributes[name] = std::make_shared<RuntimeParameterInt>(
                            static_cast<int>(attr_proto.i()));
                        break;
                    }
                    case onnx::AttributeProto_AttributeType_FLOAT:{
                        runtime_op->attributes[name] = std::make_shared<RuntimeParameterFloat>(
                            static_cast<float>(attr_proto.f()));
                        break;
                    }
                    case onnx::AttributeProto_AttributeType_INTS:{
                        std::vector<int> vals;
                        for(int64_t v:attr_proto.ints()){
                            vals.push_back(static_cast<int>(v));
                        }
                        runtime_op->attributes[name] = std::make_shared<RuntimeParameterIntArray>(
                            std::move(vals));
                        break;
                    }
                    case onnx::AttributeProto_AttributeType_FLOATS:{
                        std::vector<float> vals;
                        for(float v:attr_proto.floats()){
                            vals.push_back(v);
                        }
                        runtime_op->attributes[name] = std::make_shared<RuntimeParameterFloat>(
                            std::move(vals));
                        break;
                    }
                    case onnx::AttributeProto_AttributeType_STRING:{
                        runtime_op->attributes[name] = std::make_shared<RuntimeParameterString>(
                            attr_proto.s());
                        break;
                    }
                    default:
                        runtime_op->attributes[name] = std::make_shared<RuntimeParameter>();
                        break;
                }
            }
        }
    }
    void ONNXParser::FindGraphOutput(const onnx::GraphProto& graph_proto,RuntimeGraph &graph){
        for (const auto& output_proto : graph_proto.output()){
            const onnx::TypeProto& type_proto = output_proto.type();
            CHECK(type_proto.has_tensor_type());
            const auto& tensor_type = type_proto.tensor_type();
            const auto& shapes = tensor_type.shape();
            int type = tensor_type.elem_type();
            std::vector<uint32_t> output_shapes;
            for(int i =0;i<shapes.dim_size();++i){
                const auto& dim = shapes.dim(i);
                CHECK(dim.has_dim_value());
                output_shapes.push_back(dim.dim_value());
            }
            auto output_operand = std::unique_ptr<Operand>();
            const std::string& name = output_proto.name();
            output_operand->name = output_proto.name();
            output_operand->shapes = output_shapes;
            output_operand->data_type = MapONNXType(type);
            graph.AddOperand(name,std::move(output_operand));
        }
    }

    void ONNXParser::ParseAttributes(const onnx::GraphProto& graph_proto,RuntimeGraph &graph){

    }


}