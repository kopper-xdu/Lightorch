#include "tensor.hpp"
#include "optimizers/optimizer.hpp"
#include <vector>

template <typename Dtype>
Opt<Dtype>::Opt(std::vector<Tensor<Dtype> *> params, double lr) :
    params_(params), lr(lr) { }


template <typename Dtype>
void Opt<Dtype>::zero_grad() const
{
    // TODO
}
