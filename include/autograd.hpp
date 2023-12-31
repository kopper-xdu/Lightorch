#pragma once

#include <string>
#include <vector>
#include <functional>
#include <memory>

template<typename Dtype>
class Tensor;

template<typename Dtype>
class Functionality
{
public:
    void calc_grad(const Tensor<Dtype> &multiply_grad) const;
    std::vector<const Tensor<Dtype> *> inputs;
    std::string ops_name;

    using bwd_func = std::function<void(const std::vector<const Tensor<Dtype> *> &inputs,
                                        const std::vector<std::shared_ptr<Tensor<Dtype>>> &outputs)>;
    bwd_func ops_backward;
};