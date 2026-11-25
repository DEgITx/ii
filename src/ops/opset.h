// SPDX-License-Identifier: MIT
/// @file
/// The built-in operation set, OpSet v1. Each group registers its schemas explicitly (no static initializers).
///
/// Conventions shared by all operations:
/// - The IR is strictly typed: inputs of arithmetic operations must have the same dtype; importers insert `cast`.
/// - Binary elementwise operations broadcast with numpy rules.
/// - Negative axes count from the back.
/// - Integer division and modulo follow the definitions in `div` / `mod`.
#pragma once

#include "ir/op_schema.h"

namespace ii::ops {

/// Registers every OpSet v1 operation.
void register_opset(ir::OpRegistry& registry);

void register_elementwise_ops(ir::OpRegistry& registry);
void register_reduction_ops(ir::OpRegistry& registry);
void register_linalg_ops(ir::OpRegistry& registry);
void register_conv_ops(ir::OpRegistry& registry);
void register_image_ops(ir::OpRegistry& registry);
void register_movement_ops(ir::OpRegistry& registry);
void register_creation_ops(ir::OpRegistry& registry);
void register_quantization_ops(ir::OpRegistry& registry);
void register_composite_ops(ir::OpRegistry& registry);
/// Internal operations created by compiler passes (`fused_elementwise`); flagged `kOpInternal`.
void register_internal_ops(ir::OpRegistry& registry);

}  // namespace ii::ops
