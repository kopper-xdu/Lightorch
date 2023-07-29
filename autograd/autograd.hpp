#include "tensor.hpp"
#include <string>
#include <vector>

template<typename Dtype>
class Functionality
{
public:
    void grad();
    std::vector<Tensor<Dtype> *> inputs;
    std::string ops_name;
};