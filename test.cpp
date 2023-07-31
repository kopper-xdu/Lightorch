#include <iostream>
#include <vector>
#include "tensor.hpp"
#include <spdlog/spdlog.h>

int main() {
    Tensor<float> a({10});
    a[0] = 5;
    Tensor<float> b({10});
    b[0] = 10;
    Tensor<float> d({10});
    d[0] = 3;

    auto c = a + b;
    auto e = c * d;
    auto m = e.mean();
    std::cout << m[0] << std::endl;

    e.backward();

    std::cout << (*a.grad)[0] << std::endl;
}