#include "gtest/gtest.h"
#include "header.hpp"

TEST(OpsTest, MatMulTest) {
    Tensor<float> a({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor<float> b({3, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    std::cout << a << "\n\n" << b << '\n' << std::endl;

    auto c = a.matMul(b);

    std::cout << c << std::endl;
}