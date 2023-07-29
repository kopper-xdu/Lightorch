#include "module.hpp"
#include "tensor.hpp"

template<typename Dtype>
Linear<Dtype>::Linear(uint32_t h, uint32_t w)
    : params({h, w})
{
    
}

template<typename Dtype>
Tensor<Dtype> Linear<Dtype>::forward(Tensor<Dtype> input)
{   
    
}