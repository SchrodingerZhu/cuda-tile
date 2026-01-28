//===- AlternateReductionLoop.cpp - Alternate Reduction Loop Pass -*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"

#include "cuda_tile/Dialect/CudaTile/IR/Attributes.h"
#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
#include "cuda_tile/Dialect/CudaTile/IR/Types.h"
#include "cuda_tile/Dialect/CudaTile/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::cuda_tile;

namespace mlir::cuda_tile {

/// Check if a ForOp contains reduction operations and is a candidate
/// for alternating loop transformation.
static bool isReductionLoop(ForOp forOp) {
  bool hasReduce = false;
  bool hasMma = false;
  bool hasIterValues = forOp.getNumRegionIterVars() > 0;

  forOp.getBody()->walk([&](Operation *op) {
    if (isa<ReduceOp>(op))
      hasReduce = true;
    if (isa<MmaFOp, MmaIOp>(op))
      hasMma = true;
  });

  // A reduction loop should have iter_values (accumulators) and
  // contain either reduce operations or MMA operations
  return hasIterValues && (hasReduce || hasMma);
}

/// Transform a reduction loop into an alternating iteration pattern.
/// The original loop:
///   for j in (lb to ub, step s) { body }
/// becomes:
///   for outer_i in (0 to TILE_X, step 1) {
///     for step_idx in (lb to ub, step s) {
///       j = (outer_i % 2 == 0) ? step_idx : (ub - 1 - step_idx)
///       body using j
///     }
///   }
static void transformToAlternatingLoop(RewriterBase &rewriter, ForOp forOp,
                                        int tileXFactor) {
  Location loc = forOp.getLoc();
  Value originalLb = forOp.getLowerBound();
  Value originalUb = forOp.getUpperBound();
  Value originalStep = forOp.getStep();
  ValueRange initValues = forOp.getInitValues();

  // Get the element type for integer constants
  TileType ivType = llvm::cast<TileType>(originalLb.getType());
  Type elemType = ivType.getElementType();
  auto intType = llvm::dyn_cast<IntegerType>(elemType);
  if (!intType)
    return;

  // Create constants for the outer loop
  TileType scalarIntType = TileType::get({}, intType);
  auto createIntConst = [&](int64_t val) -> Value {
    llvm::APInt apVal(intType.getWidth(), val);
    auto constAttr = DenseIntElementsAttr::get(scalarIntType, apVal);
    return rewriter.create<ConstantOp>(loc, scalarIntType, constAttr);
  };

  Value zero = createIntConst(0);
  Value one = createIntConst(1);
  Value two = createIntConst(2);
  Value tileX = createIntConst(tileXFactor);

  // Create the outer alternating loop
  rewriter.setInsertionPoint(forOp);
  auto outerLoop = rewriter.create<ForOp>(
      loc, zero, tileX, one, initValues,
      [&](OpBuilder &builder, Location loc, Value outerIv,
          ValueRange outerIterArgs) {
        // Compute the starting bid: bid_start = original_bid * TILE_X + outer_i
        // For now, we just use outer_i as the tile index

        // Check if outer_i is even or odd: outer_i % 2
        Value remainder = builder.create<RemIOp>(loc, outerIv, two,
                                                  Signedness::Signed);
        // CmpIOp signature: (ComparisonPredicate, lhs, rhs, Signedness)
        Value isEven = builder.create<CmpIOp>(
            loc, ComparisonPredicate::EQUAL, remainder, zero,
            Signedness::Signed);

        // Create the inner loop with the same bounds as original
        auto innerLoop = builder.create<ForOp>(
            loc, originalLb, originalUb, originalStep, outerIterArgs,
            [&](OpBuilder &innerBuilder, Location innerLoc, Value stepIdx,
                ValueRange innerIterArgs) {
              // Compute the actual index j:
              // If even: j = stepIdx
              // If odd: j = (ub - 1) - stepIdx
              Value ubMinusOne =
                  innerBuilder.create<SubIOp>(innerLoc, originalUb, one);
              Value reversedIdx =
                  innerBuilder.create<SubIOp>(innerLoc, ubMinusOne, stepIdx);

              // j = isEven ? stepIdx : reversedIdx
              Value actualJ = innerBuilder.create<SelectOp>(
                  innerLoc, isEven, stepIdx, reversedIdx);

              // Clone the original loop body with the new index
              IRMapping mapper;
              mapper.map(forOp.getInductionVar(), actualJ);

              // Map the original iter args to the inner loop's iter args
              for (auto [origArg, newArg] :
                   llvm::zip(forOp.getRegionIterValues(), innerIterArgs)) {
                mapper.map(origArg, newArg);
              }

              // Clone all operations except the terminator
              SmallVector<Value> yieldValues;
              for (Operation &op : forOp.getBody()->without_terminator()) {
                innerBuilder.clone(op, mapper);
              }

              // Clone the continue operation
              auto continueOp =
                  cast<ContinueOp>(forOp.getBody()->getTerminator());
              SmallVector<Value> mappedContinueOperands;
              for (Value operand : continueOp.getOperands()) {
                mappedContinueOperands.push_back(
                    mapper.lookupOrDefault(operand));
              }
              innerBuilder.create<ContinueOp>(innerLoc, mappedContinueOperands);
            });

        // Yield the results of the inner loop to the outer loop
        builder.create<ContinueOp>(loc, innerLoop.getResults());
      });

  // Replace the original ForOp with the outer loop's results
  rewriter.replaceOp(forOp, outerLoop.getResults());
}

#define GEN_PASS_DEF_ALTERNATEREDUCTIONLOOPPASS
#include "cuda_tile/Dialect/CudaTile/Transforms/Passes.h.inc"

struct AlternateReductionLoopPass
    : public impl::AlternateReductionLoopPassBase<AlternateReductionLoopPass> {
public:
  using impl::AlternateReductionLoopPassBase<
      AlternateReductionLoopPass>::AlternateReductionLoopPassBase;

  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *ctx = &getContext();
    IRRewriter rewriter(ctx);

    // Collect ForOps to transform (to avoid invalidating iterators)
    SmallVector<ForOp> loopsToTransform;
    op->walk([&](ForOp forOp) {
      if (isReductionLoop(forOp)) {
        loopsToTransform.push_back(forOp);
      }
    });

    // Transform each collected loop
    for (ForOp forOp : loopsToTransform) {
      rewriter.setInsertionPoint(forOp);
      transformToAlternatingLoop(rewriter, forOp, tileXFactor);
    }
  }
};

} // namespace mlir::cuda_tile
