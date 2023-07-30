#include "ops/ops.hpp"
#include "tensor.hpp"
#include "common.hpp"
#include <vector>
#include <omp.h>

#define IMPLEMENT_BACKWARD(ops) \
template<typename Dtype> \
void ops##Ops<Dtype>::\
backward(const std::vector<const Tensor<Dtype> *> &inputs, \
         const std::vector<std::shared_ptr<Tensor<Dtype>>> &outputs)

template<typename Dtype>
void AddOps<Dtype>::compute(const Tensor<Dtype> &a, const Tensor<Dtype> &b, Tensor<Dtype> &out)
{
    # pragma omp parallel for
    for (int i = 0; i < a.length; ++i)
    {
        out[i] = a[i] + b[i];
    }
}

IMPLEMENT_BACKWARD(Add)
{
    auto grad_a = *(outputs[0]);
    auto grad_b = *(outputs[1]);

    # pragma omp parallel for
    for (int i = 0; i < grad_a.length; ++i)
    {
        grad_a[i] = 1;
        grad_b[i] = 1;
    }
}

template<typename Dtype>
void SubOps<Dtype>::compute(const Tensor<Dtype> &a, const Tensor<Dtype> &b, Tensor<Dtype> &out)
{
    # pragma omp parallel for
    for (int i = 0; i < a.length; ++i)
    {
        out[i] = a[i] - b[i];
    }

}

IMPLEMENT_BACKWARD(Sub)
{   
    auto grad_a = *(outputs[0]);
    auto grad_b = *(outputs[1]);

    # pragma omp parallel for
    for (int i = 0; i < grad_a.length; ++i)
    {
        grad_a[i] = 1;
        grad_b[i] = -1;
    }
}

template<typename Dtype>
void MulOps<Dtype>::compute(const Tensor<Dtype> &a, const Tensor<Dtype> &b, Tensor<Dtype> &out)
{
    # pragma omp parallel for
    for (int i = 0; i < a.length; ++i)
    {
        out[i] = a[i] * b[i];
    }

}

IMPLEMENT_BACKWARD(Mul)
{
    auto a = *(inputs[0]);
    auto b = *(inputs[1]);
    auto grad_a = *(outputs[0]);
    auto grad_b = *(outputs[1]);

    # pragma omp parallel for
    for (int i = 0; i < grad_a.length; ++i)
    {
        grad_a[i] = b[i];
        grad_b[i] = a[i];
    }
}

template<typename Dtype>
void DivOps<Dtype>::compute(const Tensor<Dtype> &a, const Tensor<Dtype> &b, Tensor<Dtype> &out)
{
    # pragma omp parallel for
    for (int i = 0; i < a.length; ++i)
    {
        out[i] = a[i] / b[i];
    }

}

IMPLEMENT_BACKWARD(Div)
{
    auto a = *(inputs[0]);
    auto b = *(inputs[1]);
    auto grad_a = *(outputs[0]);
    auto grad_b = *(outputs[1]);

    # pragma omp parallel for
    for (int i = 0; i < grad_a.length; ++i)
    {
        grad_a[i] = 1 / b[i];
        grad_b[i] = 1 / a[i];
    }
}

template<typename Dtype>
void MeanOps<Dtype>::compute(const Tensor<Dtype> &a, Tensor<Dtype> &out)
{
    Dtype sum = 0;
    # pragma omp parallel for
    for (int i = 0; i < a.length; ++i)
    {
        sum += a[i];
    }
    out[0] = sum / a.length;
}

IMPLEMENT_BACKWARD(Mean)
{
    auto a = *(inputs[0]);
    auto grad_a = *(outputs[0]);

    # pragma omp parallel for
    for (int i = 0; i < grad_a.length; ++i)
    {
        grad_a[i] = 1 / a.length;
    }
}

INSTANTIATE_CLASS(AddOps)
INSTANTIATE_CLASS(SubOps)
INSTANTIATE_CLASS(MulOps)
INSTANTIATE_CLASS(DivOps)
INSTANTIATE_CLASS(MeanOps)