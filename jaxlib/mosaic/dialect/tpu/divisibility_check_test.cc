/* Copyright 2026 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <optional>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "jaxlib/mosaic/dialect/tpu/tpu_dialect.h"

namespace mlir::tpu {
namespace {

class DivisibilityCheckTest : public ::testing::Test {
 protected:
  DivisibilityCheckTest()
      : context_([]() {
          DialectRegistry registry;
          registry.insert<arith::ArithDialect, func::FuncDialect, TPUDialect>();
          return registry;
        }()),
        builder_(&context_) {
    context_.loadAllAvailableDialects();
  }

  ~DivisibilityCheckTest() override {
    for (int i = ops_.size() - 1; i >= 0; --i) {
      ops_[i]->erase();
    }
  }

  template <typename OpTy, typename... Args>
  OpTy Create(Args&&... args) {
    OpTy op = OpTy::create(builder_, std::forward<Args>(args)...);
    ops_.push_back(op.getOperation());
    return op;
  }

  MLIRContext context_;
  OpBuilder builder_;
  std::vector<Operation*> ops_;
};

TEST_F(DivisibilityCheckTest, ConstantAddIOpIsDivisible) {
  Location loc = builder_.getUnknownLoc();
  auto c1_a = Create<arith::ConstantIndexOp>(loc, 1);
  auto c1_b = Create<arith::ConstantIndexOp>(loc, 1);

  // add = 1 + 1 = 2
  auto add = Create<arith::AddIOp>(loc, c1_a, c1_b);

  std::optional<bool> result = isDivisible(add, /*divisor=*/2);
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(*result);
}

}  // namespace
}  // namespace mlir::tpu
