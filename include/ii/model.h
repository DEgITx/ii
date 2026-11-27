// SPDX-License-Identifier: MIT
/// @file
/// Models, compiled programs and sessions.
///
/// @code
/// auto context = ii::Context::create().value();
/// auto model   = ii::Model::load(context, "model.onnx").value();  // or from_text / from_builder
/// auto program = model.compile().value();                        // optimized, backends chosen, kernels created
/// auto session = program.new_session().value();                  // one per thread / request
/// auto outputs = session.run({{"x", x}}).value();
/// @endcode
///
///   Model     the graph and its weights; immutable, shared.
///   Program   the model compiled for this machine; immutable, shared, safe to use from many threads.
///   Session   execution state (activation memory, states, bindings); use from one thread at a time.
#pragma once

#include <ii/context.h>
#include <ii/dim.h>
#include <ii/dtype.h>
#include <ii/export.h>
#include <ii/status.h>
#include <ii/tensor.h>

#include <cstddef>
#include <filesystem>
#include <initializer_list>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

II_BEGIN_DLL_EXPORTS

namespace ii {

class GraphBuilder;

namespace detail {
struct ModelImpl;
struct ProgramImpl;
struct SessionImpl;
}  // namespace detail

/// Name, element type and (possibly symbolic) shape of a model input, output or state.
struct TensorInfo {
    std::string name;
    DType dtype = DType::kInvalid;
    Dims shape;
};

/// Options of `Model::load` and `Model::from_onnx`.
struct LoadOptions {
    /// Values for symbolic dimensions of the model inputs, by name (e.g. {"batch", 1}). Fixed dimensions make shapes
    /// static, which lets the compiler evaluate more at compile time; ONNX models whose shape computations cannot be
    /// expressed symbolically (e.g. slicing a constant table by the sequence length) need them to load.
    std::vector<std::pair<std::string, std::int64_t>> dims;

    /// Inputs replaced by constants. ONNX passes some operation parameters as tensors (target shapes, axes, resize
    /// scales); when a model takes them as inputs, ii needs their values when the model is loaded.
    std::vector<std::pair<std::string, Tensor>> constant_inputs;
};

/// Options of `Model::compile`.
struct CompileOptions {
    /// Backends allowed to execute the model, in order of preference (e.g. {"ii-ref"}). Empty: all backends of the
    /// context in their default order.
    std::vector<std::string> backends;

    /// Evaluate nodes whose inputs are all constants at compile time.
    bool fold_constants = true;

    /// Merge operations: duplicates (common subexpressions), known patterns (matmul + bias into linear) and chains
    /// of elementwise operations, which ii-cpu then evaluates in one pass over memory.
    bool fuse = true;

    /// Decompose every composite operation into primitives, even when a backend implements it natively.
    /// Mostly useful for testing decompositions and backends.
    bool decompose_composites = false;

    /// Values of symbolic dimensions the planner assumes when it estimates costs, e.g. {"S", 1} to plan a language
    /// model for generating one token at a time. Other symbols count as 1.
    std::vector<std::pair<std::string, std::int64_t>> plan_dims;

    /// Memory limits in bytes for devices, by name ("cuda:0", "test:0"), replacing what the devices report. The
    /// planner moves operations to other devices to stay within them (a model larger than a GPU's memory is split
    /// between the GPU and the CPU).
    std::vector<std::pair<std::string, std::uint64_t>> device_memory;
};

class Program;
class Session;

/// Timing of one step (one kernel) of a profiled run.
struct StepProfile {
    std::string op;       ///< operation, e.g. "matmul"
    std::string name;     ///< node name (may be empty)
    std::string backend;  ///< backend executing the step, e.g. "ii-cpu"; "view" for steps that only alias memory
    std::string output;   ///< type of the first output with concrete shapes, e.g. "f32[8, 1024]"
    double start_us = 0;  ///< start relative to the beginning of the run
    double duration_us = 0;
};

/// Timings of one run of a session (see `Session::set_profiling`).
struct Profile {
    std::vector<StepProfile> steps;
    double total_us = 0;  ///< wall time of the whole run
};

/// Formats profiles as Chrome trace JSON (open in https://ui.perfetto.dev or chrome://tracing). Consecutive runs
/// are laid out one after another on the time line.
II_API std::string to_chrome_trace(std::span<const Profile> runs);

/// A model: a computation graph with its weights. Immutable; copies share the same model.
class II_API Model {
public:
    /// Takes the graph of a builder (which is left empty). Fails if the builder recorded an error.
    static Result<Model> from_builder(const Context& context, GraphBuilder&& builder);

    /// Parses a model in the ii IR text format.
    static Result<Model> from_text(const Context& context, std::string_view text);

    /// Loads a model file: ONNX (`.onnx`, external data next to it is supported) or the ii IR text format. The file
    /// is memory-mapped; suitably aligned weights are used from the mapping without copying.
    static Result<Model> load(const Context& context, const std::filesystem::path& path,
                              const LoadOptions& options = {});

    /// Imports a serialized ONNX model from memory (the data is copied; external data is not available).
    static Result<Model> from_onnx(const Context& context, std::span<const std::byte> data,
                                   const LoadOptions& options = {});

    std::vector<TensorInfo> inputs() const;
    std::vector<TensorInfo> outputs() const;
    std::vector<TensorInfo> states() const;

    /// The model in the IR text format. Constants with more than `max_constant_elements` elements are elided
    /// (`{...}`); negative prints everything.
    std::string to_text(std::int64_t max_constant_elements = 16) const;

    /// Optimizes the graph, assigns backends and creates kernels.
    Result<Program> compile(const CompileOptions& options = {}) const;

    const Context& context() const noexcept;

    /// Internal: the implementation object.
    const std::shared_ptr<const detail::ModelImpl>& impl() const noexcept { return impl_; }

private:
    explicit Model(std::shared_ptr<const detail::ModelImpl> impl) noexcept : impl_(std::move(impl)) {}
    std::shared_ptr<const detail::ModelImpl> impl_;
};

/// A model compiled for this machine. Immutable and thread-safe; create one `Session` per concurrent user.
class II_API Program {
public:
    /// Creates a session.
    Result<Session> new_session() const;

    std::vector<TensorInfo> inputs() const;
    std::vector<TensorInfo> outputs() const;

    /// Human-readable execution plan: every step with its operation, shapes and the backend executing it.
    std::string describe() const;

    /// The planner's view: devices, partitions (runs of steps on one backend) with estimated times, copies between
    /// devices, memory per device, decisions forced by memory limits and why backends declined operations.
    std::string describe_plan() const;

    /// The optimized graph in the IR text format.
    std::string to_text(std::int64_t max_constant_elements = 16) const;

    /// Internal: the implementation object.
    const std::shared_ptr<const detail::ProgramImpl>& impl() const noexcept { return impl_; }

private:
    friend class Model;
    explicit Program(std::shared_ptr<const detail::ProgramImpl> impl) noexcept : impl_(std::move(impl)) {}
    std::shared_ptr<const detail::ProgramImpl> impl_;
};

/// Execution state of a program: input/output bindings, activation memory and the values of states.
///
/// Not thread-safe: use one session per thread. In the steady state (same input shapes, same bindings) `run()`
/// performs no memory allocations.
class II_API Session {
public:
    Session(Session&&) noexcept;
    Session& operator=(Session&&) noexcept;
    ~Session();

    /// Binds an input. Contiguous host tensors are used without copying and must stay valid until the next
    /// `run()` returns; other tensors are copied.
    Status set_input(std::string_view name, const Tensor& tensor);

    /// Binds memory for an output (optional). The output is then written directly into `tensor`, which must be a
    /// contiguous host tensor with the output's dtype and the exact shape it will have.
    Status set_output(std::string_view name, const Tensor& tensor);

    /// Removes an output binding; the output is then kept in session memory again.
    Status clear_output(std::string_view name);

    /// Executes the program with the bound inputs.
    Status run();

    /// An output of the last `run()`. Unless bound with `set_output`, it refers to session memory that is reused by
    /// the next `run()` — copy it (`clone()`) to keep it.
    Result<Tensor> output(std::string_view name) const;

    /// Binds `inputs`, runs, and returns all outputs in declaration order (convenience; allocates).
    Result<std::vector<Tensor>> run(std::initializer_list<std::pair<std::string_view, Tensor>> inputs);

    /// The current value of a state (refers to session memory).
    Result<Tensor> state(std::string_view name) const;

    /// Resets all states to their initial values.
    Status reset_states();

    /// Enables or disables timing of every step in subsequent runs (costs two clock reads per step; no allocations).
    /// Steps on devices other than the host are waited for one by one, so that each time is the step's own; work
    /// that would overlap otherwise runs in sequence.
    void set_profiling(bool enabled);

    /// Step timings of the last run; fails unless profiling was enabled for it.
    Result<Profile> profile() const;

private:
    friend class Program;
    friend struct SessionTestAccess;
    explicit Session(std::unique_ptr<detail::SessionImpl> impl) noexcept;
    std::unique_ptr<detail::SessionImpl> impl_;
};

/// Outcome of one case of `check_backends`.
enum class BackendCheckState {
    kPassed,
    kFailed,   ///< wrong results, or an error other than "unsupported"
    kSkipped,  ///< the backend does not support the operation
};

struct BackendCheckCase {
    std::string name;  ///< e.g. "matmul f16"
    BackendCheckState state = BackendCheckState::kPassed;
    std::string message;   ///< why it failed or was skipped
    double max_error = 0;  ///< max |actual - expected| / (1 + |expected|)
};

/// Results of `check_backends` for one backend.
struct BackendCheck {
    std::string backend;
    std::string device;  ///< e.g. "cuda:0"
    int passed = 0;
    int failed = 0;
    int skipped = 0;
    std::vector<BackendCheckCase> cases;
};

/// A quick self-test of backends (`ii check-backends`): small models of common operations (elementwise,
/// normalizations, matmul in f32 and f16, convolution, pooling, reductions, data movement, attention, rope) run on
/// each backend alone — on a device with the copies to and from it — and are compared with ii-ref. `backends`:
/// the backends to check; empty: all but ii-ref. Meant for checking that a plugin works on a machine; the test
/// suite checks far more.
II_API Result<std::vector<BackendCheck>> check_backends(const Context& context,
                                                        std::span<const std::string> backends = {});

}  // namespace ii

II_END_DLL_EXPORTS
