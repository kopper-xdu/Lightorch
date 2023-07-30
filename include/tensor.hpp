#ifndef TENSOR_HPP_
#define TENSOR_HPP_

#include <vector>
#include <iostream>
#include <memory>
#include "storage.hpp"
#include "autograd.hpp"

template<typename Dtype>
class Functionality;

template<typename Dtype>
class Tensor
{
public:
    Tensor(const std::vector<uint32_t> &shape, bool require_grad = true);

    static Tensor ones(std::vector<uint32_t> shape);

    Dtype& operator[](uint32_t idx);
    const Dtype& operator[](uint32_t idx) const;
    Tensor operator+(const Tensor &another) const;
    Tensor operator-(const Tensor &another) const;
    Tensor operator*(const Tensor &another) const;
    Tensor operator/(const Tensor &another) const;

    void backward();

    ~Tensor();
    
    std::shared_ptr<Storage> data;
    uint32_t length = 0;
    std::vector<uint32_t> shape;

    bool require_grad = true;
    std::shared_ptr<Tensor<Dtype>> grad;
    Functionality<Dtype> grad_fn;
};  

template<typename Dtype>
std::ostream& operator<<(std::ostream& out, Tensor<Dtype> tensor);

#endif