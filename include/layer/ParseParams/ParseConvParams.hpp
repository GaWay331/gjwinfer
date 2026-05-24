#pragma once
#include "core/Operator.hpp"
#include "core/Attribute_utils.hpp"


namespace gjwinfer{
    struct ConvParams {
        //Conv的默认参数
        int stride_h = 1;
        int stride_w = 1;
        int pad_top = 0;
        int pad_left = 0;
        int pad_bottom = 0;
        int pad_right = 0;
        int dilation_h = 1;
        int dilation_w = 1;
        int group = 1;
    };
    ConvParams ParseConvParams(const Operator& op) {
        ConvParams p;
    
        auto strides = GetIntArray(op, "strides", {1, 1});
        auto pads = GetIntArray(op, "pads", {0, 0, 0, 0});
        auto dilations = GetIntArray(op, "dilations", {1, 1});
    
        CHECK_EQ(strides.size(), 2);
        CHECK_EQ(pads.size(), 4);
        CHECK_EQ(dilations.size(), 2);
    
        p.stride_h = strides[0];
        p.stride_w = strides[1];
    
        p.pad_top = pads[0];
        p.pad_left = pads[1];
        p.pad_bottom = pads[2];
        p.pad_right = pads[3];
    
        p.dilation_h = dilations[0];
        p.dilation_w = dilations[1];
    
        p.group = GetInt(op, "group", 1);
    
        return p;
    }
    
}