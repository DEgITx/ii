// SPDX-License-Identifier: MIT
/// @file
/// The backend interface.
///
/// A backend executes graph nodes on a device. Kernel backends (ii-ref, ii-cpu, ...) provide one `Kernel` per node;
/// the runtime owns scheduling and memory. Graph backends (TensorRT) compile whole subgraphs into a single `Kernel`
/// with the subgraph's inputs and outputs (`Backend::compiles_subgraphs`).
///
/// A backend runs on one device (`device.h`). The planner chooses a backend for every node from the estimates of
/// the backends (`estimate_us`) and the cost of moving values between devices.
///
/// Kernels are created once per compiled `Program`, are immutable and shared by all sessions; `run()` receives
/// concrete shapes and memory and must not allocate (P7) — it may use the workspace it requested instead.
#pragma once

#include <ii/dtype.h>
#include <ii/executor.h>
#include <ii/shape.h>
#include <ii/status.h>
#include <ii/tensor.h>

#include "backend/device.h"
#include "ir/graph.h"
#include "ir/op_schema.h"

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>

namespace ii::backend {

/// A dense, row-major tensor argument of a kernel. An absent optional input has `data == nullptr` and
/// `dtype == DType::kInvalid`.
struct TensorView {
    void* data = nullptr;
    DType dtype = DType::kInvalid;
    Shape shape;

    bool present() const noexcept { return dtype != DType::kInvalid; }
    std::int64_t numel() const noexcept { return shape.numel(); }
    std::size_t nbytes() const noexcept { return storage_size(dtype, numel()); }
    int rank() const noexcept { return shape.rank(); }

    template <class T>
    T* as() const noexcept {
        return static_cast<T*>(data);
    }
};

/// Arguments of one kernel execution.
struct KernelContext {
    std::span<const TensorView> inputs;
    std::span<const TensorView> outputs;
    std::span<std::byte> workspace;  ///< 64-byte aligned scratch memory of the requested size, on the device
    Executor* executor = nullptr;    ///< for parallel work; never null during execution
    Queue* queue = nullptr;          ///< kernels of a device other than the host enqueue their work here

    /// Input `index`; an absent view for omitted trailing optional inputs (a node may have fewer inputs than its
    /// schema declares).
    const TensorView& input(int index) const noexcept {
        static constexpr TensorView kAbsent{};
        return static_cast<std::size_t>(index) < inputs.size() ? inputs[static_cast<std::size_t>(index)] : kAbsent;
    }
    const TensorView& output(int index = 0) const noexcept { return outputs[static_cast<std::size_t>(index)]; }
};

/// An executable implementation of one node (or of a fused subgraph).
class Kernel {
public:
    virtual ~Kernel() = default;

    /// Scratch memory needed for the given concrete shapes. Called when shapes are (re)bound, never during `run`.
    virtual std::size_t workspace_size(std::span<const Shape> inputs, std::span<const Shape> outputs) const {
        (void)inputs;
        (void)outputs;
        return 0;
    }

    /// Executes the kernel. Must be thread-safe (sessions may run concurrently) and must not allocate.
    virtual Status run(const KernelContext& context) const = 0;

    /// True if output 0 may share memory with input `input` when both have the same shape and element size and the
    /// input is not needed afterwards (the runtime decides; the kernel must then work with `output.data ==
    /// input.data`). Elementwise kernels that read each element before writing the same position return true.
    virtual bool can_run_in_place(int input) const {
        (void)input;
        return false;
    }

    /// True if the kernel copied constant input `input` when it was created (into its own packed layout or device
    /// memory) and never reads it at run time. The runtime then does not copy that constant to the kernel's device;
    /// the input pointer passed to `run` may point to another device.
    virtual bool captures_input(int input) const {
        (void)input;
        return false;
    }

    /// True if `run` only enqueues work on the device queue (`Queue::begin_recording`) whose arguments depend on
    /// nothing but the shapes and pointers it gets: no reads of device data on the host, no waits, no host memory.
    /// The runtime may then record the work once and replay it in later runs without calling `run`.
    virtual bool can_record() const { return false; }

protected:
    Kernel() = default;
    Kernel(const Kernel&) = default;
    Kernel& operator=(const Kernel&) = default;
};

/// An execution backend.
class Backend {
public:
    virtual ~Backend() = default;

    /// Short unique name, e.g. "ii-ref".
    virtual std::string_view name() const noexcept = 0;

    /// The device whose memory the kernels read and write, and its performance figures.
    virtual const DeviceInfo& device_info() const noexcept = 0;
    Device device() const noexcept { return device_info().device; }

    /// Memory and queues of the device; nullptr for host backends. Backends on the same device may return the same
    /// object; the runtime uses the first one.
    virtual DeviceApi* device_api() const noexcept { return nullptr; }

    /// Estimated execution time of a supported node with the given concrete shapes, in microseconds. The default is
    /// the roofline of `device_info()`: max(flops / peak, bytes / bandwidth) + launch overhead; 0 for views the
    /// runtime does not execute (`is_free_view`).
    virtual double estimate_us(const ir::CostContext& context) const;

    /// OK if the backend can execute the node; otherwise an `kUnsupported` status explaining why (shown by
    /// `ii inspect --plan`).
    virtual Status supports(const ir::Node& node) const = 0;

    /// Creates the kernel for a supported node.
    virtual Result<std::unique_ptr<Kernel>> create_kernel(const ir::Node& node) const = 0;

    /// True for graph backends (TensorRT): besides single nodes they compile `subgraph` nodes (ops/fused.h), parts
    /// of the graph handed to them after planning (runtime/partition.h). Their `supports` and `estimate_us` answer
    /// for single nodes (so that the planner can place nodes on their device) and for `subgraph` nodes, whose estimate
    /// decides whether the part goes to the backend; a node the planner gave them is always part of a subgraph.
    virtual bool compiles_subgraphs() const { return false; }

protected:
    Backend() = default;
    Backend(const Backend&) = default;
    Backend& operator=(const Backend&) = default;
};

/// True if permuting `shape` by `perm` keeps the order of the elements (it only moves dimensions of size 1, e.g.
/// [1, S, H, D] -> [1, H, S, D] with S = 1): the runtime then runs the permute as a view of its input.
bool keeps_element_order(std::span<const std::int64_t> perm, const Shape& shape);

/// True for nodes the runtime turns into views of their input instead of running them: view operations (reshape)
/// and permutes that keep the element order.
bool is_free_view(const ir::CostContext& context);

/// A backend built from a table of per-operation kernel factories.
class KernelBackend : public Backend {
public:
    using Factory = Result<std::unique_ptr<Kernel>> (*)(const ir::Node& node);
    using SupportCheck = Status (*)(const ir::Node& node);

    Status supports(const ir::Node& node) const override;
    Result<std::unique_ptr<Kernel>> create_kernel(const ir::Node& node) const override;

    /// Registers the kernel of an operation. `check` (optional) refines support (dtypes, attributes).
    void add_kernel(std::string_view op, Factory factory, SupportCheck check = nullptr);

    /// True when a kernel is registered for `op`.
    bool has_kernel(std::string_view op) const noexcept { return kernels_.contains(op); }

private:
    struct Entry {
        Factory factory;
        SupportCheck check;
    };
    std::unordered_map<std::string_view, Entry> kernels_;  // keys point into the static OpSchema names
};

}  // namespace ii::backend
