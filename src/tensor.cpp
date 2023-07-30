#include "tensor.hpp"
#include "ops/ops.hpp"
#include "common.hpp"

template<typename Dtype>
Tensor<Dtype>::Tensor(const std::vector<uint32_t> &shape, bool require_grad) :
    shape(shape), 
    require_grad(require_grad)
    // data(new Storage())
{
    auto mem_size = sizeof(Dtype);
    for (auto x : shape)
    {
        mem_size *= x;
    }
    this->length = mem_size / sizeof(Dtype);
    this->data = std::make_shared<Storage>(mem_size);
    if (require_grad)
    {
        this->grad = std::make_shared<Tensor<Dtype>>(shape, false);
    }
}

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::ones(std::vector<uint32_t> shape)
{
    // Tensor
}

template<typename Dtype>
void Tensor<Dtype>::backward()
{
    // this->grad->data;
    (*(this->grad))[0] = 1;
    this->grad_fn.calc_grad(*(this->grad));
}

template<typename Dtype>
Tensor<Dtype>::~Tensor()
{
    // free(data);
}

template<typename Dtype>
Dtype& Tensor<Dtype>::operator[](uint32_t idx)
{
    return ((Dtype *) this->data->data)[idx];
}

template<typename Dtype>
const Dtype& Tensor<Dtype>::operator[](uint32_t idx) const
{
    return ((Dtype *) this->data->data)[idx];
}

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::operator+(const Tensor &another) const
{
    Tensor<Dtype> res(another.shape);
    AddOps<Dtype>::compute(*this, another, res);

    res.grad_fn.inputs.push_back(this);
    res.grad_fn.inputs.push_back(&another);
    res.grad_fn.ops_name = "Add";
    res.grad_fn.ops_backward = &AddOps<Dtype>::backward;

    return res;
}

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::operator-(const Tensor &another) const
{
    Tensor<Dtype> res(another.shape);
    SubOps<Dtype>::compute(*this, another, res);

    res.grad_fn.inputs.push_back(this);
    res.grad_fn.inputs.push_back(&another);
    res.grad_fn.ops_name = "Sub";
    res.grad_fn.ops_backward = &SubOps<Dtype>::backward;

    return res;
}

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::operator*(const Tensor &another) const
{
    Tensor<Dtype> res(another.shape);
    MulOps<Dtype>::compute(*this, another, res);

    res.grad_fn.inputs.push_back(this);
    res.grad_fn.inputs.push_back(&another);
    res.grad_fn.ops_name = "Mul";
    res.grad_fn.ops_backward = &MulOps<Dtype>::backward;

    return res;
}

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::operator/(const Tensor &another) const
{
    Tensor<Dtype> res(another.shape);
    DivOps<Dtype>::compute(*this, another, res);

    res.grad_fn.inputs.push_back(this);
    res.grad_fn.inputs.push_back(&another);
    res.grad_fn.ops_name = "Div";
    res.grad_fn.ops_backward = &DivOps<Dtype>::backward;

    return res;
}

template<typename Dtype>
std::ostream& operator<<(std::ostream& out, Tensor<Dtype> tensor)
{
    return out;
}

INSTANTIATE_CLASS(Tensor)