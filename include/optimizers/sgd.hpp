#pragma once

#include "tensor.hpp"
#include <vector>
#include "optimizers/optimizer.hpp"


template <typename Dtype>
class SGD : Opt<Dtype> {
public:
    SGD(std::vector<Tensor<Dtype> *> params, double lr) : Opt(params, lr) { }
};