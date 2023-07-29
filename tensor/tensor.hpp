#ifndef TENSOR
#define TENSOR

#include <vector>
#include <iostream>
#include "autograd.hpp"

template<typename Dtype>
class Tensor
{
public:
    Tensor(std::vector<uint32_t> shape);

    Dtype operator[](uint32_t idx);
    Tensor operator+(const Tensor &another);
    Tensor operator-(const Tensor &another);
    Tensor operator*(const Tensor &another);
    Tensor operator/(const Tensor &another);

    void backward();

    ~Tensor();
    
    Dtype *data = nullptr;
    uint32_t length = 0;
    std::vector<int> shape;

    // bool require_grad = true;
    Tensor grad;
    Functionality<Dtype> grad_fn;
    
};  

template<typename Dtype>
std::ostream& operator<<(std::ostream& out, Tensor<Dtype> tensor);

#endif