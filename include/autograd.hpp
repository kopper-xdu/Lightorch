#ifndef AUTOGRAD
#define AUTOGRAD

// #include "tensor.hpp"
#include <string>
#include <vector>

template<typename Dtype>
class Tensor;

template<typename Dtype>
class Functionality
{
public:
    void calc_grad(const Tensor<Dtype> &multiply_grad) const;
    std::vector<const Tensor<Dtype> *> inputs;
    std::string ops_name;

    using bwd_ptr = void (*)(const Tensor<Dtype> &a, const Tensor<Dtype> &b, Tensor<Dtype> &grad_a, Tensor<Dtype> &grad_b);
    bwd_ptr ops_backward;
};

#endif