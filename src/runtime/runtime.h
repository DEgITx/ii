// SPDX-License-Identifier: MIT
/// @file
/// Implementation objects behind the public `Context`, `Model`, `Program` and `Session` handles.
#pragma once

#include <ii/context.h>
#include <ii/model.h>

#include "backend/backend.h"
#include "core/thread_pool.h"
#include "ir/graph.h"
#include "runtime/planner.h"

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ii::detail {

/// Page-locked host memory of the context's devices (`DeviceApi::allocate_host`), or the context's allocator when no
/// device has any (`Context::pinned_allocator`).
class PinnedAllocator final : public Allocator {
public:
    PinnedAllocator(Allocator& fallback, std::vector<backend::DeviceApi*> devices)
        : fallback_(fallback), devices_(std::move(devices)) {}

    void* allocate(std::size_t size, std::size_t alignment) noexcept override;
    void deallocate(void* ptr, std::size_t size, std::size_t alignment) noexcept override;
    /// True if `ptr` was allocated from a device's page-locked memory.
    bool is_pinned(const void* ptr) noexcept;

private:
    Allocator& fallback_;
    std::vector<backend::DeviceApi*> devices_;
    std::mutex mutex_;
    std::unordered_map<void*, std::shared_ptr<Buffer>> blocks_;
};

struct ContextImpl {
    Allocator* allocator = nullptr;
    Executor* executor = nullptr;
    std::unique_ptr<ThreadPool> own_pool;
    Logger logger;
    std::vector<std::unique_ptr<backend::Backend>> backends;  // priority order
    std::vector<PluginDescription> plugins;
    // Declared after the backends: destroyed before them.
    std::unique_ptr<PinnedAllocator> pinned;
    std::mutex copy_mutex;  ///< guards copy_queues and their use
    std::vector<std::pair<backend::DeviceApi*, std::unique_ptr<backend::Queue>>> copy_queues;  ///< Context::copy

    const backend::Backend* find_backend(std::string_view name) const noexcept;
    /// The reference backend (always present); used for constant folding.
    const backend::Backend& reference() const;
    /// The memory of a device other than the host: that of its first backend that executes kernels (ii-cuda rather
    /// than TensorRT), as programs use it; nullptr if the context has no such device.
    backend::DeviceApi* device_api(Device device) const noexcept;
};

/// A device a program runs on. Device 0 of a program is always the host.
struct ProgramDevice {
    backend::DeviceInfo info;
    backend::DeviceApi* api = nullptr;  ///< nullptr for the host
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
    int device = 0;        ///< the program device whose memory holds the value
};

/// One kernel invocation, or a copy between devices (`transfer`, executed by the runtime).
struct Step {
    const ir::Node* node = nullptr;
    const backend::Backend* backend = nullptr;  ///< nullptr for transfers
    std::unique_ptr<backend::Kernel> kernel;    ///< nullptr for transfers
    std::vector<int> inputs;                    ///< value indices; -1 for absent optional inputs
    std::vector<int> outputs;                   ///< value indices
    bool symbolic_attrs = false;  ///< attributes contain symbolic dimensions (substituted when shapes are bound)
    int device = 0;               ///< program device executing the kernel (transfers: the destination)
    bool transfer = false;
    int transfer_from = 0;   ///< transfers: source device
    double estimate_us = 0;  ///< the planner's estimate (kernels)

    bool is_transfer() const noexcept { return transfer; }
};

struct StateInfo {
    std::string name;
    int value = -1;   ///< the state value
    int update = -1;  ///< the value written back; -1 if never updated
    Tensor initial;   ///< empty: zeros
    int device = 0;   ///< the program device holding the state
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
    std::vector<ProgramDevice> devices;  ///< [0] is the host
    runtime::GraphPlan plan;             ///< the planner's estimates and decisions (for `describe`)

    int find_input(std::string_view name) const noexcept;
    int find_output(std::string_view name) const noexcept;
    int find_state(std::string_view name) const noexcept;
};

/// Concrete shapes, memory layout and pre-built kernel arguments for one binding of the symbols.
struct ExecutionPlan {
    std::vector<std::pair<std::string, std::int64_t>> symbols;  ///< bound symbol values
    std::vector<Shape> input_shapes;                            ///< the input shapes the plan was built for
    std::vector<Shape> shapes;                                  ///< per value
    std::vector<std::size_t> offsets;                ///< per value: offset in the arena of its device, or kNotInArena
    std::vector<std::size_t> workspace_offsets;      ///< per step, in the arena of the step's device
    std::vector<std::size_t> workspace_sizes;        ///< per step
    std::vector<std::size_t> state_staging_offsets;  ///< per state: staging buffer on its device, or kNotInArena
    std::vector<char> alias_steps;  ///< per step: the output is the input's memory (a view), the kernel is skipped
    /// Per value: the state whose memory holds it (an in-place update of that state, e.g. a KV cache write), or -1.
    std::vector<int> state_alias;
    std::vector<std::size_t> arena_sizes;    ///< per program device
    std::vector<backend::TensorView> views;  ///< per step: inputs then outputs
    std::vector<std::size_t> view_begin;     ///< per step, plus a final end entry

    /// A run of steps on one device whose kernels can be recorded (`Kernel::can_record`; views in between are only
    /// pointer bookkeeping). When a run repeats the kernel arguments of the previous one, the queue records the
    /// steps once (a CUDA graph) and later runs replay the recording instead of calling the kernels.
    struct Segment {
        std::size_t begin = 0;  ///< first step
        std::size_t end = 0;    ///< one past the last step
        int device = 0;
        std::vector<const void*> pointers;              ///< kernel arguments of the last run
        int repeats = 0;                                ///< runs in a row with `pointers`
        std::unique_ptr<backend::Recording> recording;  ///< recorded with `pointers`
        bool disabled = false;                          ///< the queue or a kernel could not record
    };
    std::vector<Segment> segments;
    std::vector<int> segment_at;  ///< per step: the segment starting there, or -1

    static constexpr std::size_t kNotInArena = static_cast<std::size_t>(-1);
};

struct SessionImpl {
    std::shared_ptr<const ProgramImpl> program;
    std::vector<Tensor> inputs;                           ///< bound inputs (contiguous)
    std::vector<Tensor> bound_outputs;                    ///< optional user output buffers
    std::vector<Tensor> states;                           ///< current state values
    std::vector<Shape> state_shapes;                      ///< shapes the states were allocated for
    std::vector<std::shared_ptr<Buffer>> arenas;          ///< per program device
    std::vector<std::unique_ptr<backend::Queue>> queues;  ///< per program device; nullptr for the host
    std::vector<std::byte*> bases;                        ///< per program device: arena base of the current run
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

    std::vector<const void*> segment_pointers;  ///< scratch of run_segment

    // Memory of other devices and page-locked memory.
    bool host_transfers = false;  ///< the program copies between the host and devices
    bool pinned_arena = false;    ///< the host arena is page-locked: copies from it run asynchronously
    /// Per input: the program device whose transfer steps are the input's only readers (an input bound in that
    /// device's memory is used there directly), or -1.
    std::vector<int> input_direct_device;
    std::vector<std::vector<int>> input_transfer_steps;  ///< per input: the transfer steps reading it
    std::vector<int> input_devices;                      ///< per input: program device of the bound tensor
    std::vector<int> output_devices;                     ///< per output: program device of the bound tensor
    /// Per output: copied to the host by a transfer step and read by no other step (an output bound in the source
    /// device's memory is copied there instead).
    std::vector<char> output_direct;
    /// Per step, set by each run: 1 a transfer to a device whose input is already there (skipped), 2 a transfer to
    /// the host whose output is bound in the source device's memory (a copy within the device).
    std::vector<std::uint8_t> transfer_overrides;
    /// Per program device: a copy from the page-locked host arena may still be reading its source; host steps that
    /// may overwrite it wait for the device first.
    std::vector<std::uint8_t> pending_reads;
    bool any_pending_reads = false;
    Status wait_for_pending_reads();

    Status prepare();  ///< builds or reuses the execution plan for the bound input shapes
    Status run();
    Status execute();  ///< the steps of `run()`; queues may hold work when it fails
    Status run_step(std::size_t step, backend::KernelContext& context);
    /// Replays (or records and replays) a segment; false when its steps must run one by one.
    Result<bool> run_segment(ExecutionPlan::Segment& segment, backend::KernelContext& context);
    Status synchronize();  ///< waits for the work of every device queue
    Result<Tensor> value_tensor(int value) const;
    Status reset_states();
    Status allocate_states();
    Status initialize_state(std::size_t state);  ///< sets a state to its initial value (zeros when it has none)
    Allocator& host_allocator() const;  ///< for host copies of inputs: page-locked if the program copies to devices
};

}  // namespace ii::detail
