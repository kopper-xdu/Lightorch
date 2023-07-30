#include "autograd.hpp"
#include "ops/ops.hpp"
#include "tensor.hpp"
#include "common.hpp"
#include <vector>

template<typename Dtype>
void Functionality<Dtype>::calc_grad(const Tensor<Dtype> &multiply_grad) const
{
    if (this->inputs.size() == 0)
        return ;
    this->ops_backward(*(this->inputs[0]), *(this->inputs[1]), *(this->inputs[0]->grad), *(this->inputs[1]->grad));

    for (auto input : this->inputs)
    {
        *(input->grad) = *(input->grad) * multiply_grad;
        input->grad_fn.calc_grad(*(input->grad));
    }

}

INSTANTIATE_CLASS(Functionality)