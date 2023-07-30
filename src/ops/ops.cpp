#include "ops/ops.hpp"
#include "tensor.hpp"
#include "common.hpp"
#include <vector>


template<typename Dtype>
void AddOps<Dtype>::compute(const Tensor<Dtype> &a, const Tensor<Dtype> &b, Tensor<Dtype> &out)
{
    for (int i = 0; i < a.length; ++i)
    {
        out[i] = a[i] + b[i];
    }

}

template<typename Dtype>
void AddOps<Dtype>::backward(const Tensor<Dtype> &a, const Tensor<Dtype> &b, Tensor<Dtype> &grad_a, Tensor<Dtype> &grad_b)
{
    for (int i = 0; i < grad_a.length; ++i)
    {
        grad_a[i] = 1;
        grad_b[i] = 1;
    }
}

template<typename Dtype>
void SubOps<Dtype>::compute(const Tensor<Dtype> &a, const Tensor<Dtype> &b, Tensor<Dtype> &out)
{

    for (int i = 0; i < a.length; ++i)
    {
        out[i] = a[i] - b[i];
    }

}

template<typename Dtype>
void SubOps<Dtype>::backward(const Tensor<Dtype> &a, const Tensor<Dtype> &b, Tensor<Dtype> &grad_a, Tensor<Dtype> &grad_b)
{
    for (int i = 0; i < grad_a.length; ++i)
    {
        grad_a[i] = 1;
        grad_b[i] = -1;
    }
}

template<typename Dtype>
void MulOps<Dtype>::compute(const Tensor<Dtype> &a, const Tensor<Dtype> &b, Tensor<Dtype> &out)
{

    for (int i = 0; i < a.length; ++i)
    {
        out[i] = a[i] * b[i];
    }

}

template<typename Dtype>
void MulOps<Dtype>::backward(const Tensor<Dtype> &a, const Tensor<Dtype> &b, Tensor<Dtype> &grad_a, Tensor<Dtype> &grad_b)
{

    for (int i = 0; i < grad_a.length; ++i)
    {
        grad_a[i] = b[i];
        grad_b[i] = a[i];
    }
}

template<typename Dtype>
void DivOps<Dtype>::compute(const Tensor<Dtype> &a, const Tensor<Dtype> &b, Tensor<Dtype> &out)
{

    for (int i = 0; i < a.length; ++i)
    {
        out[i] = a[i] / b[i];
    }

}

template<typename Dtype>
void DivOps<Dtype>::backward(const Tensor<Dtype> &a, const Tensor<Dtype> &b, Tensor<Dtype> &grad_a, Tensor<Dtype> &grad_b)
{

    for (int i = 0; i < grad_a.length; ++i)
    {
        grad_a[i] = 1 / b[i];
        grad_b[i] = 1 / a[i];
    }
}

INSTANTIATE_CLASS(AddOps)
INSTANTIATE_CLASS(SubOps)
INSTANTIATE_CLASS(MulOps)
INSTANTIATE_CLASS(DivOps)