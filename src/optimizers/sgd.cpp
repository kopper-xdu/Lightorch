#include "tensor.hpp"
#include <vector>
#include "optimizers/optimizer.hpp"
#include "optimizers/sgd.hpp"


template <typename Dtype>
SGD<Dtype>::SGD(std::vector<Tensor<Dtype> *> params, double lr) : Opt(params, lr) { }

