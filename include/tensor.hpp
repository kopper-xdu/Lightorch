#pragma once

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
    Tensor(const std::vector<uint32_t> &shape, bool require_grad = true, bool alloc_mem = true);
    Tensor(const std::vector<uint32_t> &shape, const std::initializer_list<Dtype> &init_vals, bool require_grad = true, bool alloc_mem = true);

    // Tensor slice(const std::vector<std::pair<uint32_t, uint32_t>>& ranges) const;
    Dtype item() const;
    Tensor operator=(Dtype val) const;
    Tensor operator[](uint32_t idx) const;
    Tensor operator+(const Tensor &another) const;
    Tensor operator-(const Tensor &another) const;
    Tensor operator*(const Tensor &another) const;
    Tensor operator/(const Tensor &another) const;
    Tensor mean() const;
    Tensor matMul(const Tensor &another) const;


    void backward();

    ~Tensor();
    
    std::shared_ptr<Storage> data_;
    uint32_t length_ = 0;
    std::vector<uint32_t> shape_;
    std::vector<uint32_t> stride_;
    uint32_t data_offset_;

    bool require_grad_ = true;
    std::shared_ptr<Tensor<Dtype>> grad_;
    Functionality<Dtype> grad_fn_;
};  

template<typename Dtype>
std::ostream& operator<<(std::ostream& out, const Tensor<Dtype> &tensor);
