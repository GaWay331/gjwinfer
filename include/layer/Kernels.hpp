#pragma once


namespace gjwinfer{
class ExecutionKernel{
public:
    virtual ~ExecutionKernel() = default;

    virtual void Init() = 0;

    virtual void InferShape() = 0;

    virtual void Forward() = 0;
};
}