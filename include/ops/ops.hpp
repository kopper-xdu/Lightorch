#ifndef OPS
#define OPS

#include <vector>
#include <map>
#include <memory>

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

#define DECLARE_BACKWARD \
static void backward(const std::vector<const Tensor<Dtype> *> &inputs, \
                     const std::vector<std::shared_ptr<Tensor<Dtype>>> &outputs);

template<typename Dtype>
class AddOps
{
public:
    static void compute(const Tensor<Dtype> &a, const Tensor<Dtype> &b, Tensor<Dtype> &out);
    DECLARE_BACKWARD
};

template<typename Dtype>
class SubOps
{
public:
    static void compute(const Tensor<Dtype> &a, const Tensor<Dtype> &b, Tensor<Dtype> &out);
    DECLARE_BACKWARD
};

template<typename Dtype>
class MulOps
{
public:
    static void compute(const Tensor<Dtype> &a, const Tensor<Dtype> &b, Tensor<Dtype> &out);
    DECLARE_BACKWARD
};

template<typename Dtype>
class DivOps
{
public:
    static void compute(const Tensor<Dtype> &a, const Tensor<Dtype> &b, Tensor<Dtype> &out);
    DECLARE_BACKWARD
};

template<typename Dtype>
class MeanOps
{
public:
    static void compute(const Tensor<Dtype> &a, Tensor<Dtype> &out);
    DECLARE_BACKWARD
};


#endif