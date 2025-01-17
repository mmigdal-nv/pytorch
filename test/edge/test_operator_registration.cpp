#include "operator_registry.h"
#include <gtest/gtest.h>

namespace torch {
namespace executor {

// add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
TEST(OperatorRegistrationTest, Add) {
    EValue values[4];
    values[0] = EValue(at::ones({2, 3}));
    values[1] = EValue(at::ones({2, 3}));
    values[2] = EValue(int64_t(1));
    values[3] = EValue(at::zeros({2, 3}));
    ASSERT_TRUE(hasOpsFn("aten::add.out"));
    auto op = getOpsFn("aten::add.out");

    EValue* kernel_values[4];
    for (size_t i = 0; i < 4; i++) {
        kernel_values[i] = &values[i];
    }
    op(kernel_values);
    at::Tensor expected = at::ones({2, 3});
    expected = at::fill(expected, 2);
    ASSERT_TRUE(expected.equal(kernel_values[3]->toTensor()));

}
} // namespace executor
} // namespace torch
