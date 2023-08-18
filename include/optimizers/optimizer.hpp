#pragma once

#include "tensor.hpp"
#include <vector>


template <typename Dtype>
class Opt {
public:
    Opt(std::vector<Tensor<Dtype> *> params, double lr)

    virtual void step() const;
    void zero_grad() const;

    std::vector<Tensor<Dtype> *> params_;
    double lr;
};
