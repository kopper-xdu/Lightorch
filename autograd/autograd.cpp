#include "autograd.hpp"
#include "ops.hpp"
#include <vector>

template<typename Dtype>
void Functionality<Dtype>::grad()
{
    std::vector<Tensor<Dtype> *> grads
    for (auto input : this->inputs)
    {
        grads.push_back(&(input->grad));
    }
    AddOps::backward(this->inputs, grads);

    
}