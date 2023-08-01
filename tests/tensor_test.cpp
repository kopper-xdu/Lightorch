#include "gtest/gtest.h"
#include "header.hpp"
#include <iostream>

TEST(TensorTest, BasicOpsTest) {
    Tensor<float> a({1});
    a[0] = 5;
    Tensor<float> b({1});
    b[0] = 10;
    Tensor<float> c({1});
    c[0] = 3;

    auto d = a + b;
    auto g = a - b;
    auto e = d * c;
    auto h = e / c;
    auto f = e.mean();

    EXPECT_EQ(d[0], 15);
    EXPECT_EQ(e[0], 45);
    EXPECT_EQ(f[0], 45);
    EXPECT_EQ(g[0], -5);
    EXPECT_EQ(h[0], 15);

    // Tensor<float> aa({10, 5});
    // std::cout << aa << std::endl;
}

TEST(TensorTest, BwdOpsTest) {
    Tensor<float> a({1});
    a[0] = 5;
    Tensor<float> b({1});
    b[0] = 10;
    Tensor<float> c({1});
    c[0] = 3;

    auto d = a + b;
    auto e = d * c;
    auto f = e.mean();

    f.backward();

    EXPECT_EQ((*a.grad_)[0], 3);
    EXPECT_EQ((*b.grad_)[0], 3);
    EXPECT_EQ((*c.grad_)[0], 15);
    EXPECT_EQ((*d.grad_)[0], 3);
    EXPECT_EQ((*e.grad_)[0], 1);
    EXPECT_EQ((*f.grad_)[0], 1);
}
