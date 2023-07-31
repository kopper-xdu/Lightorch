#include "gtest/gtest.h"
#include "header.hpp"

TEST(TensorTest, BasicOpsTest) {
    Tensor<float> a({1});
    a[0] = 5;
    Tensor<float> b({1});
    b[0] = 10;
    Tensor<float> c({1});
    c[0] = 3;

    auto k = a + b;
    auto e = k * c;
    auto m = e.mean();

    EXPECT_EQ(k[0], 15);
    EXPECT_EQ(e[0], 45);
    EXPECT_EQ(m[0], 45);
}
