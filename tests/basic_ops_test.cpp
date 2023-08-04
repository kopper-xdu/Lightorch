#include "gtest/gtest.h"
#include "header.hpp"
#include <iostream>

TEST(TensorTest, BasicOpsTest) {
    Tensor<float> a({1});
    a = 5;
    Tensor<float> b({1});
    b = 10;
    Tensor<float> c({1});
    c = 3;

    auto d = a + b;
    auto g = a - b;
    auto e = d * c;
    auto h = e / c;
    auto f = e.mean();

    EXPECT_EQ(d[0].item(), 15);
    EXPECT_EQ(e[0].item(), 45);
    EXPECT_EQ(f[0].item(), 45);
    EXPECT_EQ(g[0].item(), -5);
    EXPECT_EQ(h[0].item(), 15);
}

TEST(TensorTest, BasicBwdOpsTest) {
    Tensor<float> a({1});
    a = 5;
    Tensor<float> b({1});
    b = 10;
    Tensor<float> c({1});
    c = 3;

    auto d = a + b;
    auto e = d * c;
    auto f = e.mean();

    f.backward();

    EXPECT_EQ((*a.grad_)[0].item(), 3);
    EXPECT_EQ((*b.grad_)[0].item(), 3);
    EXPECT_EQ((*c.grad_)[0].item(), 15);
    EXPECT_EQ((*d.grad_)[0].item(), 3);
    EXPECT_EQ((*e.grad_)[0].item(), 1);
    EXPECT_EQ((*f.grad_)[0].item(), 1);
}

// TEST(TensorTest, OutTest) {
//     Tensor<float> a({10}, {1, 2, 3, 4, 5, 6});
//     Tensor<float> b({5, 5}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
//     Tensor<float> c({2, 4, 2, 3});

//     std::cout << a << '\n' << std::endl;
//     std::cout << b << '\n' << std::endl;
//     std::cout << c << '\n' << std::endl;
// }
