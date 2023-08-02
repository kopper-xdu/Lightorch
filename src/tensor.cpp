#include "tensor.hpp"
#include "ops/ops.hpp"
#include "common.hpp"
#include <string>

uint32_t compute_length(const std::vector<uint32_t> &shape)
{
    if (shape.size() == 0)
        return 0;

    uint32_t length = 1;
    for (auto x : shape)
    {
        length *= x;
    }
    return length;
}

template<typename Dtype>
Tensor<Dtype>::Tensor(const std::vector<uint32_t> &shape, bool require_grad, bool alloc_mem) :
    shape_(shape), 
    require_grad_(require_grad),
    length_(compute_length(shape)),
    stride_(shape.size()),
    data_offset_(0)
{
    if (alloc_mem)
        data_ = std::make_shared<Storage>(length_ * sizeof(Dtype));
    if (require_grad_)
    {
        grad_ = std::make_shared<Tensor<Dtype>>(shape_, false);
    }

    uint32_t stride = 1;
    for (int i = shape.size() - 1; i >= 0; i--) {
        stride_[i] = stride;
        stride *= shape_[i];
    }
}

template<typename Dtype>
void Tensor<Dtype>::backward()
{
    auto ptr = (Dtype *) (grad_->data_->start_ptr_);
    ptr[0] = 1;
    grad_fn_.calc_grad(*grad_);
}

template<typename Dtype>
Tensor<Dtype>::~Tensor()
{
    // free(data);
}

// template<typename Dtype>
// Tensor<Dtype> Tensor<Dtype>::slice(const std::vector<std::pair<uint32_t, uint32_t>> &ranges) const {
//     std::vector<uint32_t> shape;
//     uint32_t length = 1;

//     for (uint32_t i = 0; i < ranges.size(); i++) {
//         uint32_t start = ranges[i].first;
//         uint32_t end = ranges[i].second;

//         if (start < 0) {
//             start += shape_[i];
//         }
//         if (end < 0) {
//             end += shape_[i];
//         }

//         uint32_t dim_size = end - start;
//         shape.push_back(dim_size);
//         length *= dim_size;
//     }

//     for (int i = ranges.size(); i < shape_.size(); ++i) {
//         shape.push_back(shape_[i]);
//         length *= shape_[i];
//     }

//     Tensor<Dtype> res(shape);
//     SliceOps<Dtype>::compute({this, }, res);

//     res.grad_fn_.ops_backward = std::bind(SliceOps<Dtype>::backward, 
//                                      std::placeholders::_1, 
//                                      std::placeholders::_2
//                                      );

//     res.grad_fn.inputs.push_back(this);
//     res.grad_fn.ops_name = "Slice";
//     res.grad_fn.ops_backward = &SliceOps<Dtype>::backward;

//     return res;
// }

template<typename Dtype>
Dtype Tensor<Dtype>::item() const
{
    return *(Dtype *) data_->start_ptr_;
}

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::operator[](uint32_t idx) const
{
    // return ((Dtype *) data_->start_ptr_)[idx];
    auto rank = shape_.size();

    std::vector<uint32_t> shape;
    if (rank > 1)
        shape = std::vector<uint32_t>(shape_.begin() + 1, shape_.end());
    else
        shape = std::vector<uint32_t>({1});

    Tensor res(shape, true, false);
    res.data_ = data_;
    res.data_offset_ += idx * stride_[0];

    return res;
}

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::operator=(Dtype val) const
{
    // memset(data_->start_ptr_, va)
    Dtype* ptr = (Dtype*) data_->start_ptr_;
    for (size_t i = 0; i < length_; ++i) {
        memcpy(ptr + i, &val, sizeof(Dtype));
    }
    return *this;
}

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::operator+(const Tensor &another) const
{
    Tensor<Dtype> res(another.shape_);
    AddOps<Dtype>::compute({this, &another}, res);

    res.grad_fn_.ops_backward = std::bind(AddOps<Dtype>::backward, 
                                     std::placeholders::_1, 
                                     std::placeholders::_2
                                     );

    res.grad_fn_.inputs.push_back(this);
    res.grad_fn_.inputs.push_back(&another);
    res.grad_fn_.ops_name = "Add";
    // res.grad_fn.ops_backward = &AddOps<Dtype>::backward;

    return res;
}

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::operator-(const Tensor &another) const
{
    Tensor<Dtype> res(another.shape_);
    SubOps<Dtype>::compute({this, &another}, res);

    res.grad_fn_.ops_backward = std::bind(SubOps<Dtype>::backward, 
                                     std::placeholders::_1, 
                                     std::placeholders::_2
                                     );

    res.grad_fn_.inputs.push_back(this);
    res.grad_fn_.inputs.push_back(&another);
    res.grad_fn_.ops_name = "Sub";
    // res.grad_fn.ops_backward = &SubOps<Dtype>::backward;

    return res;
}

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::operator*(const Tensor &another) const
{
    Tensor<Dtype> res(another.shape_);
    MulOps<Dtype>::compute({this, &another}, res);

    res.grad_fn_.ops_backward = std::bind(MulOps<Dtype>::backward, 
                                     std::placeholders::_1, 
                                     std::placeholders::_2
                                     );

    res.grad_fn_.inputs.push_back(this);
    res.grad_fn_.inputs.push_back(&another);
    res.grad_fn_.ops_name = "Mul";
    // res.grad_fn.ops_backward = &MulOps<Dtype>::backward;

    return res;
}

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::operator/(const Tensor &another) const
{
    Tensor<Dtype> res(another.shape_);
    DivOps<Dtype>::compute({this, &another}, res);

    res.grad_fn_.ops_backward = std::bind(DivOps<Dtype>::backward, 
                                     std::placeholders::_1, 
                                     std::placeholders::_2
                                     );
    res.grad_fn_.inputs.push_back(this);
    res.grad_fn_.inputs.push_back(&another);
    res.grad_fn_.ops_name = "Div";
    // res.grad_fn.ops_backward = &DivOps<Dtype>::backward;

    return res;
}

template<typename Dtype>
Tensor<Dtype> Tensor<Dtype>::mean() const
{
    Tensor<Dtype> res({1});
    MeanOps<Dtype>::compute({this, }, res);

    res.grad_fn_.ops_backward = std::bind(MeanOps<Dtype>::backward, 
                                     std::placeholders::_1, 
                                     std::placeholders::_2
                                     );

    res.grad_fn_.inputs.push_back(this);
    res.grad_fn_.ops_name = "Mean";
    // res.grad_fn.ops_backward = &MeanOps<Dtype>::backward;

    return res;
}

template<typename Dtype>
std::ostream& operator<<(std::ostream& out, const Tensor<Dtype> &tensor)
{
    auto &shape = tensor.shape_;
    int rank = shape.size();

    if (rank == 0) {
        out << "tensor([])";
    } else if (rank == 1) {
        out << "[ ";
        for (int i = 0; i < shape[0]; i++) {
            out << ((Dtype *) tensor.data_->start_ptr_)[i];
            if (i < shape[0] - 1) {
                out << ", ";
            }
        }
        out << " ]";
    } else {
        out << "[ ";
        for (int i = 0; i < shape[0]; i++) {
            Tensor<Dtype> slice = tensor[i];
            out << slice;
            if (i < shape[0] - 1) {
                out << ",\n";
                if (rank > 2)
                    out << '\n';
            }
        }
        out << " ]";
        
    }

    return out;
}

INSTANTIATE_CLASS(Tensor)
template std::ostream& operator<<(std::ostream& out, const Tensor<float> &tensor);
template std::ostream& operator<<(std::ostream& out, const Tensor<double> &tensor);