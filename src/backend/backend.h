// SPDX-License-Identifier: MIT
/// @file
/// The backend interface.
///
/// A backend executes graph nodes on a device. Kernel backends (ii-ref, ii-cpu, ...) provide one `Kernel` per node;
/// the runtime owns scheduling and memory. Graph backends (TensorRT, ...; later stages) compile whole subgraphs into
/// a single `Kernel` with the subgraph's inputs and outputs.
///
/// Kernels are created once per compiled `Program`, are immutable and shared by all sessions; `run()` receives
/// concrete shapes and memory and must not allocate (P7) — it may use the workspace it requested instead.
#pragma once

#include <ii/dtype.h>
#include <ii/executor.h>
#include <ii/shape.h>
#include <ii/status.h>
#include <ii/tensor.h>

#include "ir/graph.h"

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
    std::span<std::byte> workspace;  ///< 64-byte aligned scratch memory of the requested size
    Executor* executor = nullptr;    ///< for parallel work; never null during execution

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

    /// The device whose memory the kernels read and write.
    virtual Device device() const noexcept { return Device::cpu(); }

    /// OK if the backend can execute the node; otherwise an `kUnsupported` status explaining why (shown by
    /// `ii inspect --plan`).
    virtual Status supports(const ir::Node& node) const = 0;

    /// Creates the kernel for a supported node.
    virtual Result<std::unique_ptr<Kernel>> create_kernel(const ir::Node& node) const = 0;

protected:
    Backend() = default;
    Backend(const Backend&) = default;
    Backend& operator=(const Backend&) = default;
};

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
