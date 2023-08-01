#include "autograd.hpp"
#include "ops/ops.hpp"
#include "tensor.hpp"
#include "common.hpp"
#include <vector>
#include <iostream>

template<typename Dtype>
void Functionality<Dtype>::calc_grad(const Tensor<Dtype> &multiply_grad) const
{
    if (inputs.size() == 0)
        return ;
    // this->ops_backward(*(this->inputs[0]), *(this->inputs[1]), *(this->inputs[0]->grad), *(this->inputs[1]->grad));
    
    std::vector<std::shared_ptr<Tensor<Dtype>>> outputs;
    for (auto input : inputs)
    {
        outputs.push_back(input->grad_);
    }

    // std::forward(inputs, outputs);
    // std::forward(outputs);
    ops_backward(inputs, outputs);

    for (auto input : inputs)
    {
        *(input->grad_) = *(input->grad_) * multiply_grad;
        input->grad_fn_.calc_grad(*(input->grad_));
    }

}

INSTANTIATE_CLASS(Functionality)