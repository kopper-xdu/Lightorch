#include "tensor.hpp"
#include <vector>


template<typename Dtype>
class Module
{
public:
    virtual Tensor<Dtype> forward(Tensor<Dtype> input) = 0;
    Tensor<Dtype> operator()(Tensor<Dtype> input) { this->forward(input); }
    virtual ~Module() = 0;

    Tensor<Dtype> params;
};

template<typename Dtype>
class Linear : public Module<Dtype>
{
public:
    Linear(uint32_t h, uint32_t w);
    virtual Tensor<Dtype> forward(Tensor<Dtype> input);
    virtual ~Linear();
};

