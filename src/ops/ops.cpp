#include "ops/ops.hpp"
#include "tensor.hpp"
#include "common.hpp"
#include <vector>
#include <omp.h>
#include "utils/timer.hpp"

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

IMPLEMENT_COMPUTE(MatMul)
{
    auto a = (Dtype *) inputs[0]->data_->start_ptr_;
    auto b = (Dtype *) inputs[1]->data_->start_ptr_;
    auto o = (Dtype *) out.data_->start_ptr_;

    memset(o, 0, sizeof(Dtype) * out.length_);

    // do not support batch
    auto rows1 = inputs[0]->shape_[0];
    auto cols1 = inputs[0]->shape_[1];

    auto rows2 = inputs[1]->shape_[0];
    auto cols2 = inputs[1]->shape_[1];

    Timer timer("MatMul_fwd");
    # pragma omp parallel for
    for (int i = 0; i < rows1; ++i) {
        for (int j = 0; j < cols2; ++j) {
            for (int k = 0; k < cols1; ++k) {
                auto v = a[i * cols1 + k] * b[k * cols2 + j];
                o[i * cols2 + j] += a[i * cols1 + k] * b[k * cols2 + j];
            }
        }
    }
}

IMPLEMENT_BACKWARD(MatMul)
{
    // auto a = (Dtype *) inputs[0]->data_->start_ptr_;
    // auto b = (Dtype *) inputs[1]->data_->start_ptr_;
    // auto o = (Dtype *) out.data_->start_ptr_;

    // memset(o, 0, sizeof(Dtype) * out.length_);

    // // do not support batch
    // auto rows1 = inputs[0]->shape_[0];
    // auto cols1 = inputs[0]->shape_[1];

    // auto rows2 = inputs[1]->shape_[0];
    // auto cols2 = inputs[1]->shape_[1];

    // Timer timer("MatMul_fwd");
    // # pragma omp parallel for
    // for (int i = 0; i < rows1; ++i) {
    //     for (int j = 0; j < cols2; ++j) {
    //         for (int k = 0; k < cols1; ++k) {
    //             o[i * rows1 + j] += a[i * rows1 + k] * b[k * rows2 + j];
    //         }
    //     }
    // }
}

INSTANTIATE_CLASS(AddOps)
INSTANTIATE_CLASS(SubOps)
INSTANTIATE_CLASS(MulOps)
INSTANTIATE_CLASS(DivOps)
INSTANTIATE_CLASS(MeanOps)
INSTANTIATE_CLASS(MatMulOps)


// IMPLEMENT_BACKWARD(MatMul)
// {
//     auto a = (Dtype *) inputs[0]->data_->start_ptr_;
//     auto b = (Dtype *) inputs[1]->data_->start_ptr_;
//     auto grad_a = (Dtype *) outputs[0]->data_->start_ptr_;
//     auto grad_b = (Dtype *) outputs[1]->data_->start_ptr_;

//     // uint32_t length = outputs[0]->length_;
//     // do not support batch
//     auto lrows = inputs[0]->shape_[0];
//     auto lcols = inputs[0]->shape_[1];

//     Timer timer("MatMul_bwd");
//     # pragma omp parallel for
//     for (int i = 0; i < rows; ++i) {
//         for (int j = 0; j < cols; ++j) {
//             for (int k = 0; k < lcols; ++k) {

//             }
//         }
//     }
// }

// template<typename Dtype>
// void SliceOps<Dtype>::compute(const std::vector<const Tensor<Dtype> *> &inputs,
//                               Tensor<Dtype> &out,
//                               const std::vector<std::pair<uint32_t, uint32_t>> &ranges)
// {
//     auto a = (Dtype *) inputs[0]->data_->start_ptr_;
//     auto o = (Dtype *) out.data_->start_ptr_;

//     uint32_t length = out->length_;

//     # pragma omp parallel for
//     for (int i = 0; i < length; ++i)
//     {
//         // o[i] = ;
//     }
// }

// template<typename Dtype>
// void SliceOps<Dtype>::backward(const std::vector<const Tensor<Dtype> *> &inputs,
//                                const std::vector<std::shared_ptr<Tensor<Dtype>>> &outputs,
//                                const std::vector<std::pair<uint32_t, uint32_t>> ranges)
// {
// }