#include "ops/ops.hpp"
#include "tensor.hpp"
#include "common.hpp"
#include <vector>
#include <omp.h>

#define IMPLEMENT_COMPUTE(ops) \
template<typename Dtype> \
void ops##Ops<Dtype>::\
compute(const std::vector<const Tensor<Dtype> *> &inputs, \
        Tensor<Dtype> &out)

#define IMPLEMENT_BACKWARD(ops) \
template<typename Dtype> \
void ops##Ops<Dtype>::\
backward(const std::vector<const Tensor<Dtype> *> &inputs, \
         const std::vector<std::shared_ptr<Tensor<Dtype>>> &outputs)

IMPLEMENT_COMPUTE(Add)
{
    auto a = (Dtype *) inputs[0]->data_->start_ptr_;
    auto b = (Dtype *) inputs[1]->data_->start_ptr_;
    auto o = (Dtype *) out.data_->start_ptr_;

    uint32_t length = inputs[0]->length_;

    # pragma omp parallel for
    for (int i = 0; i < length; ++i)
    {
        o[i] = a[i] + b[i];
    }
}

IMPLEMENT_BACKWARD(Add)
{
    auto grad_a = (Dtype *) outputs[0]->data_->start_ptr_;
    auto grad_b = (Dtype *) outputs[1]->data_->start_ptr_;

    uint32_t length = outputs[0]->length_;

    # pragma omp parallel for
    for (int i = 0; i < length; ++i)
    {
        grad_a[i] = 1;
        grad_b[i] = 1;
    }
}

IMPLEMENT_COMPUTE(Sub)
{
    auto a = (Dtype *) inputs[0]->data_->start_ptr_;
    auto b = (Dtype *) inputs[1]->data_->start_ptr_;
    auto o = (Dtype *) out.data_->start_ptr_;

    uint32_t length = inputs[0]->length_;

    # pragma omp parallel for
    for (int i = 0; i < length; ++i)
    {
        o[i] = a[i] - b[i];
    }
}

IMPLEMENT_BACKWARD(Sub)
{   
    auto grad_a = (Dtype *) outputs[0]->data_->start_ptr_;
    auto grad_b = (Dtype *) outputs[1]->data_->start_ptr_;

    uint32_t length = outputs[0]->length_;

    # pragma omp parallel for
    for (int i = 0; i < length; ++i)
    {
        grad_a[i] = 1;
        grad_b[i] = -1;
    }
}

IMPLEMENT_COMPUTE(Mul)
{
    auto a = (Dtype *) inputs[0]->data_->start_ptr_;
    auto b = (Dtype *) inputs[1]->data_->start_ptr_;
    auto o = (Dtype *) out.data_->start_ptr_;

    uint32_t length = inputs[0]->length_;

    # pragma omp parallel for
    for (int i = 0; i < length; ++i)
    {
        o[i] = a[i] * b[i];
    }

}

IMPLEMENT_BACKWARD(Mul)
{
    auto a = (Dtype *) inputs[0]->data_->start_ptr_;
    auto b = (Dtype *) inputs[1]->data_->start_ptr_;
    auto grad_a = (Dtype *) outputs[0]->data_->start_ptr_;
    auto grad_b = (Dtype *) outputs[1]->data_->start_ptr_;

    uint32_t length = outputs[0]->length_;

    # pragma omp parallel for
    for (int i = 0; i < length; ++i)
    {
        grad_a[i] = b[i];
        grad_b[i] = a[i];
    }
}

IMPLEMENT_COMPUTE(Div)
{
    auto a = (Dtype *) inputs[0]->data_->start_ptr_;
    auto b = (Dtype *) inputs[1]->data_->start_ptr_;
    auto o = (Dtype *) out.data_->start_ptr_;

    uint32_t length = inputs[0]->length_;

    # pragma omp parallel for
    for (int i = 0; i < length; ++i)
    {
        o[i] = a[i] / b[i];
    }

}

IMPLEMENT_BACKWARD(Div)
{
    auto a = (Dtype *) inputs[0]->data_->start_ptr_;
    auto b = (Dtype *) inputs[1]->data_->start_ptr_;
    auto grad_a = (Dtype *) outputs[0]->data_->start_ptr_;
    auto grad_b = (Dtype *) outputs[1]->data_->start_ptr_;

    uint32_t length = outputs[0]->length_;

    # pragma omp parallel for
    for (int i = 0; i < length; ++i)
    {
        grad_a[i] = 1 / b[i];
        grad_b[i] = 1 / a[i];
    }
}

IMPLEMENT_COMPUTE(Mean)
{
    auto a = (Dtype *) inputs[0]->data_->start_ptr_;
    auto o = (Dtype *) out.data_->start_ptr_;

    Dtype sum = 0;
    uint32_t length = inputs[0]->length_;

    # pragma omp parallel for
    for (int i = 0; i < length; ++i)
    {
        sum += a[i];
    }

    o[0] = sum / length;
}

IMPLEMENT_BACKWARD(Mean)
{
    auto a = (Dtype *) inputs[0]->data_->start_ptr_;
    auto grad_a = (Dtype *) outputs[0]->data_->start_ptr_;

    uint32_t length = inputs[0]->length_;
    # pragma omp parallel for
    for (int i = 0; i < length; ++i)
    {
        grad_a[i] = 1 / length;
    }
}

template<typename Dtype>
void SliceOps<Dtype>::compute(const std::vector<const Tensor<Dtype> *> &inputs,
                              Tensor<Dtype> &out,
                              const std::vector<std::pair<uint32_t, uint32_t>> &ranges)
{
    auto a = (Dtype *) inputs[0]->data_->start_ptr_;
    auto o = (Dtype *) out.data_->start_ptr_;

    uint32_t length = out->length_;

    # pragma omp parallel for
    for (int i = 0; i < length; ++i)
    {
        // o[i] = ;
    }
}

template<typename Dtype>
void SliceOps<Dtype>::backward(const std::vector<const Tensor<Dtype> *> &inputs,
                               const std::vector<std::shared_ptr<Tensor<Dtype>>> &outputs,
                               const std::vector<std::pair<uint32_t, uint32_t>> ranges)
{
}

INSTANTIATE_CLASS(AddOps)
INSTANTIATE_CLASS(SubOps)
INSTANTIATE_CLASS(MulOps)
INSTANTIATE_CLASS(DivOps)
INSTANTIATE_CLASS(MeanOps)