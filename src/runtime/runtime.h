// SPDX-License-Identifier: MIT
/// @file
/// Implementation objects behind the public `Context`, `Model`, `Program` and `Session` handles.
#pragma once

#include <ii/context.h>
#include <ii/model.h>

#include "backend/backend.h"
#include "core/thread_pool.h"
#include "ir/graph.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace ii::detail {

struct ContextImpl {
    Allocator* allocator = nullptr;
    Executor* executor = nullptr;
    std::unique_ptr<ThreadPool> own_pool;
    Logger logger;
    std::vector<std::unique_ptr<backend::Backend>> backends;  // priority order

    const backend::Backend* find_backend(std::string_view name) const noexcept;
    /// The reference backend (always present); used for constant folding.
    const backend::Backend& reference() const;
};

struct ModelImpl {
    explicit ModelImpl(Context owner) : context(std::move(owner)) {}
    Context context;
    std::unique_ptr<ir::Graph> graph;
};

/// A value of the compiled program.
struct ValueInfo {
    ir::ValueKind kind = ir::ValueKind::kNode;
    ir::TensorType type;
    std::string name;
    Tensor constant;       ///< kConstant
    int input_index = -1;  ///< kInput
    int state_index = -1;  ///< kState
    int producer = -1;     ///< step producing the value (kNode)
    int last_use = -1;     ///< last step reading the value; `steps.size()` when it is a graph result
};

/// One kernel invocation.
struct Step {
    const ir::Node* node = nullptr;
    const backend::Backend* backend = nullptr;
    std::unique_ptr<backend::Kernel> kernel;
    std::vector<int> inputs;      ///< value indices; -1 for absent optional inputs
    std::vector<int> outputs;     ///< value indices
    bool symbolic_attrs = false;  ///< attributes contain symbolic dimensions (substituted when shapes are bound)
};

struct StateInfo {
    std::string name;
    int value = -1;   ///< the state value
    int update = -1;  ///< the value written back; -1 if never updated
    Tensor initial;   ///< empty: zeros
};

struct ProgramImpl {
    explicit ProgramImpl(Context owner) : context(std::move(owner)) {}
    Context context;
    std::unique_ptr<ir::Graph> graph;  ///< the optimized graph; steps point into it
    std::vector<ValueInfo> values;
    std::vector<Step> steps;
    std::vector<int> inputs;  ///< value index per graph input
    std::vector<std::string> input_names;
    std::vector<int> outputs;  ///< value index per graph output
    std::vector<std::string> output_names;
    std::vector<StateInfo> states;

    int find_input(std::string_view name) const noexcept;
    int find_output(std::string_view name) const noexcept;
    int find_state(std::string_view name) const noexcept;
};

/// Concrete shapes, memory layout and pre-built kernel arguments for one binding of the symbols.
struct ExecutionPlan {
    std::vector<std::pair<std::string, std::int64_t>> symbols;  ///< bound symbol values
    std::vector<Shape> input_shapes;                            ///< the input shapes the plan was built for
    std::vector<Shape> shapes;                                  ///< per value
    std::vector<std::size_t> offsets;                           ///< per value: arena offset, or kNotInArena
    std::vector<std::size_t> workspace_offsets;                 ///< per step
    std::vector<std::size_t> workspace_sizes;                   ///< per step
    std::vector<std::size_t> state_staging_offsets;             ///< per state: staging buffer, or kNotInArena
    std::vector<char> alias_steps;  ///< per step: the output is the input's memory (a view), the kernel is skipped
    /// Per value: the state whose memory holds it (an in-place update of that state, e.g. a KV cache write), or -1.
    std::vector<int> state_alias;
    std::size_t arena_size = 0;
    std::vector<backend::TensorView> views;  ///< per step: inputs then outputs
    std::vector<std::size_t> view_begin;     ///< per step, plus a final end entry

    static constexpr std::size_t kNotInArena = static_cast<std::size_t>(-1);
};

struct SessionImpl {
    std::shared_ptr<const ProgramImpl> program;
    std::vector<Tensor> inputs;         ///< bound inputs (contiguous)
    std::vector<Tensor> bound_outputs;  ///< optional user output buffers
    std::vector<Tensor> states;         ///< current state values
    std::vector<Shape> state_shapes;    ///< shapes the states were allocated for
    std::shared_ptr<Buffer> arena;
    ExecutionPlan plan;
    bool planned = false;
    /// Recently used plans for other input shapes (an LLM alternates prompt and single-token shapes).
    std::vector<ExecutionPlan> plan_cache;
    static constexpr std::size_t kPlanCacheSize = 4;
    bool has_run = false;
    std::vector<void*> value_data;  ///< per value, rebuilt by every run (no allocation)
    bool profiling = false;
    bool profiled = false;                 ///< the last run was profiled
    std::vector<std::int64_t> step_times;  ///< per step: start and end in ns since the start of the run
    std::int64_t run_time_ns = 0;

    Status prepare();  ///< builds or reuses the execution plan for the bound input shapes
    Status run();
    Result<Tensor> value_tensor(int value) const;
    Status reset_states();
    Status allocate_states();
};

}  // namespace ii::detail
