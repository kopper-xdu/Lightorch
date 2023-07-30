#include <iostream>
#include "tensor.hpp"
#include <vector>


int main()
{
    Tensor<float> a({1});
    a[0] = 5;
    Tensor<float> b({1});
    b[0] = 10;
    Tensor<float> d({1});
    d[0] = 3;

    auto c = a + b;
    auto e = c * d;
    std::cout << e[0] << std::endl;

    e.backward();

    std::cout << (*a.grad)[0] << std::endl;
}