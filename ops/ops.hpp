#include "tensor.hpp"
#include <vector>

template<typename Dtype>
class Ops
{
// public:
//     virtual Tensor<Dtype> compute(Tensor<Dtype> a, Tensor<Dtype> b) = 0;
//     virtual Tensor<Dtype> grad() = 0;
};

template<typename Dtype>
class AddOps : public Ops
{
public:
    void compute(Tensor<Dtype> &a, Tensor<Dtype> &b, Tensor<Dtype> &out);
    void backward(std::vector<Tensor<Dtype> *> &input, std::vector<Tensor<Dtype> *> &out);
};

template<typename Dtype>
class SubOps : public Ops
{
public:
    Tensor<Dtype> compute(Tensor<Dtype> &a, Tensor<Dtype> &b, Tensor<Dtype> &out);
    Tensor<Dtype> backward();
};