#include "ops.hpp"
#include "tensor.hpp"
#include <vector>


template<typename Dtype>
void AddOps<Dtype>::compute(Tensor<Dtype> &a, Tensor<Dtype> &b, Tensor<Dtype> &out)
{
    for (int i = 0; i < this->length; ++i)
    {
        out.data[i] = a.data[i] + b.data[i];
    }

}

template<typename Dtype>
void AddOps<Dtype>::backward(std::vector<Tensor<Dtype> *> &input, std::vector<Tensor<Dtype> *> &out)
{

}