#include "tensor.hpp"
#include <ops.hpp>

template<typename Dtype>
Tensor<Dtype>::Tensor(std::vector<uint32_t> shape)
    : shape(shape)
{
    auto mem_size = sizeof(Dtype);
    for (auto x : shape)
    {
        mem_size *= x;
    }
    length = mem_size / sizeof(Dtype);
    data = (Dtype *) malloc(mem_size);
}

template<typename Dtype>
void Tensor<Dtype>::backward()
{
    this->grad_fn.grad();
}

template<typename Dtype>
Tensor<Dtype>::~Tensor()
{
    free(data);
}

template<typename Dtype>
Dtype Tensor<Dtype>::operator[](uint32_t idx)
{
    return this->data[idx];
}

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::operator+(const Tensor &another)
{
    Tensor<Dtype> res(another.shape);
    AddOps::compute(*this, another, res);

    res.grad_fn.inputs.push_back(this);
    res.grad_fn.inputs.push_back(&another);
    res.grad_fn.ops_name = "add";
    return res;
}

template<typename Dtype>
std::ostream& operator<<(std::ostream& out, Tensor<Dtype> tensor)
{
    return out;
}
