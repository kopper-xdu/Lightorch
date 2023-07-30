#ifndef OPS
#define OPS

#include <vector>
#include <map>

template<typename Dtype>
class Tensor;

// template<typename Dtype>
// class Ops
// {
// public:
//     // virtual Tensor<Dtype> compute(Tensor<Dtype> a, Tensor<Dtype> b) = 0;
//     // virtual Tensor<Dtype> grad() = 0;
//     void blank()
// };

template<typename Dtype>
class AddOps
{
public:
    static void compute(const Tensor<Dtype> &a, const Tensor<Dtype> &b, Tensor<Dtype> &out);
    static void backward(const Tensor<Dtype> &a, const Tensor<Dtype> &b, Tensor<Dtype> &grad_a, Tensor<Dtype> &grad_b);
};

template<typename Dtype>
class SubOps
{
public:
    static void compute(const Tensor<Dtype> &a, const Tensor<Dtype> &b, Tensor<Dtype> &out);
    static void backward(const Tensor<Dtype> &a, const Tensor<Dtype> &b, Tensor<Dtype> &grad_a, Tensor<Dtype> &grad_b);
};

template<typename Dtype>
class MulOps
{
public:
    static void compute(const Tensor<Dtype> &a, const Tensor<Dtype> &b, Tensor<Dtype> &out);
    static void backward(const Tensor<Dtype> &a, const Tensor<Dtype> &b, Tensor<Dtype> &grad_a, Tensor<Dtype> &grad_b);
};

template<typename Dtype>
class DivOps
{
public:
    static void compute(const Tensor<Dtype> &a, const Tensor<Dtype> &b, Tensor<Dtype> &out);
    static void backward(const Tensor<Dtype> &a, const Tensor<Dtype> &b, Tensor<Dtype> &grad_a, Tensor<Dtype> &grad_b);
};


#endif