# pragma once

#include <vector>
#include <array>
#include <memory>

template<typename Dtype>
class Tensor;

#define DECLARE_COMPUTE(num) \
static void compute(const std::vector<const Tensor<Dtype> *> &inputs, \
                    Tensor<Dtype> &out);

#define DECLARE_BACKWARD(num) \
static void backward(const std::vector<const Tensor<Dtype> *> &inputs, \
                     const std::vector<std::shared_ptr<Tensor<Dtype>>> &outputs);

#define DECLARE_OPS(ops_name, num) \
template<typename Dtype> \
class ops_name##Ops \
{ \
public: \
    DECLARE_COMPUTE(num) \
    DECLARE_BACKWARD(num) \
};

DECLARE_OPS(Add, 2)
DECLARE_OPS(Sub, 2)
DECLARE_OPS(Mul, 2)
DECLARE_OPS(Div, 2)
DECLARE_OPS(Mean, 1)


template<typename Dtype>
class SliceOps
{
public:
    DECLARE_COMPUTE(1)
    DECLARE_BACKWARD(1)

    const std::vector<std::pair<uint32_t, uint32_t>> ranges;
};