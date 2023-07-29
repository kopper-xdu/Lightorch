#include "tensor.hpp"

int main()
{
    Tensor<int> a({3, 4});
    Tensor<int> b({3, 4});
    auto c = a + b;
    
}