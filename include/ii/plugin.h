/* SPDX-License-Identifier: MIT */
/**
 * @file
 * The binary interface between ii and backend plugins (`ii_backend_plugin_v1`).
 *
 * A backend plugin is a shared library (`ii-backend-<name>.dll`, `libii-backend-<name>.so`,
 * `libii-backend-<name>.dylib`) that adds backends to ii: ii-cuda, TensorRT, or a backend of your own. ii loads it
 * at run time, so a program linked with ii works on machines without the plugin's SDK (CUDA, TensorRT, ...), and the
 * plugin may be built with another compiler than ii (MSVC and MinGW, say): only the C declarations below cross the
 * boundary. Plugins are found in `ContextOptions::plugin_paths`, in the directories listed in the `II_PLUGIN_PATH`
 * environment variable, and next to the ii library (see docs/guides/backend-plugins.md).
 *
 * The plugin exports one function, `ii_backend_plugin_v1`, which returns a table of functions. ii calls
 * `create_backends` once per `ii::Context`; each backend then answers queries about graph nodes (`supports`,
 * `estimate_us`) and creates kernels (`create_kernel`) which ii runs (`run`). Backends of a device other than the host
 * also provide its memory (`allocate`, `deallocate`) and queues of ordered work (`create_queue`, the copy functions,
 * `synchronize`) — the semantics are those of `ii::backend::Kernel`, `DeviceApi` and `Queue` (src/backend/).
 *
 * Graph nodes are passed as a graph in the ii IR text format (src/ir/text.h) with one node: its inputs are graph
 * inputs (`%x0`, `%x1`, ...) or constants (`%c0`, ...) whose data is elided (`{...}`) and passed by pointer in
 * `ii_plugin_graph::constants`; its outputs are graph outputs. Later versions may pass larger subgraphs to the same
 * functions. A plugin built with the ii sources parses the text with `ii::ir::parse_graph` — the plugin SDK
 * (src/plugin/sdk.h) does this and adapts a C++ `ii::backend::Backend` to this interface.
 *
 * Rules:
 *   - Functions returning `int32_t` return a status code (`ii_status` values: 0 is success); on failure they also
 *     store the code and a NUL-terminated message (truncated to fit) in `*error`.
 *   - Strings are UTF-8 and NUL-terminated. Strings returned by the plugin stay valid while the object they describe
 *     lives; strings passed by ii are valid only during the call.
 *   - Element types are `ii::DType` values (`ii_dtype` in ii/c_api.h, plus block-quantized types); device kinds are
 *     `ii::DeviceKind` values. Both only grow, by appending.
 *   - Compatibility: ii loads plugins whose `abi_version` equals `II_PLUGIN_ABI_VERSION`. New functions are appended
 *     to the tables; `struct_size` tells which exist. A plugin must not unwind exceptions through this interface.
 *   - Thread safety: `supports`, `estimate_us`, `create_kernel`, `workspace_size`, `run`, `allocate` and
 *     `deallocate` may be called concurrently; a queue is used by one thread at a time.
 */
#ifndef II_PLUGIN_H
#define II_PLUGIN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* C declarations: typedef is intended. NOLINTBEGIN(modernize-use-using) */

/** Version of this interface. A breaking change gets a new entry point (`ii_backend_plugin_v2`). */
#define II_PLUGIN_ABI_VERSION 1

/** Name of the function a plugin exports: `const ii_plugin_v1* ii_backend_plugin_v1(uint32_t host_abi_version)`. */
#define II_PLUGIN_ENTRY_POINT "ii_backend_plugin_v1"

#define II_PLUGIN_MAX_RANK 8
#define II_PLUGIN_ERROR_SIZE 1024

#if defined(_WIN32) || defined(__CYGWIN__)
#define II_PLUGIN_EXPORT __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#define II_PLUGIN_EXPORT __attribute__((visibility("default")))
#else
#define II_PLUGIN_EXPORT
#endif

/** Status code and message of a failed call. */
typedef struct ii_plugin_error {
    int32_t code;
    char message[II_PLUGIN_ERROR_SIZE];
} ii_plugin_error;

/** Log levels (`ii::LogLevel` values). */
enum {
    II_PLUGIN_LOG_TRACE = 0,
    II_PLUGIN_LOG_DEBUG = 1,
    II_PLUGIN_LOG_INFO = 2,
    II_PLUGIN_LOG_WARNING = 3,
    II_PLUGIN_LOG_ERROR = 4
};

/** A dense, row-major tensor. An absent optional input has `dtype == 0` and `data == NULL`. */
typedef struct ii_plugin_tensor {
    void* data;
    int32_t dtype;
    int32_t rank;
    int64_t shape[II_PLUGIN_MAX_RANK];
} ii_plugin_tensor;

/** A concrete shape (`rank == -1`: an absent input). */
typedef struct ii_plugin_shape {
    int32_t rank;
    int32_t reserved;
    int64_t dims[II_PLUGIN_MAX_RANK];
} ii_plugin_shape;

/** Data of a constant of a graph, by the name it has in the text (without '%'). */
typedef struct ii_plugin_constant {
    const char* name;
    const void* data; /**< dense, row-major, in host memory */
    size_t size;      /**< bytes */
} ii_plugin_constant;

/**
 * A graph in the ii IR text format with the data of its constants. In `create_kernel` the constant data stays valid
 * while the kernel lives; elsewhere only during the call.
 */
typedef struct ii_plugin_graph {
    const char* text;
    size_t text_size;
    const ii_plugin_constant* constants;
    int32_t constant_count;
} ii_plugin_graph;

/** Capacity and speed of a backend's device: the inputs of ii's planner (see `ii::backend::DeviceInfo`). */
typedef struct ii_plugin_device_info {
    int32_t kind;          /**< `ii::DeviceKind`: 0 the host, 1 CUDA, 2 the simulated test device */
    int32_t index;         /**< device index within its kind */
    const char* name;      /**< e.g. "NVIDIA GeForce RTX 5080" */
    uint64_t memory_bytes; /**< memory ii may use; 0: unlimited */
    double gflops;         /**< peak arithmetic throughput */
    double bandwidth_gbs;  /**< memory bandwidth */
    double launch_us;      /**< fixed cost of one kernel */
    double link_gbs;       /**< host <-> device copies */
    double link_latency_us;
} ii_plugin_device_info;

/** The executor (thread pool) of the context, for kernels that run on the host. */
typedef struct ii_plugin_executor {
    void* context;
    int32_t concurrency;
    /** Calls `fn(user, begin, end)` for disjoint subranges covering [0, n), possibly concurrently; see `ii::Executor`.
     */
    void (*parallel_for)(void* context, int64_t n, int64_t grain, void (*fn)(void* user, int64_t begin, int64_t end),
                         void* user);
} ii_plugin_executor;

typedef struct ii_plugin_backend ii_plugin_backend;     /**< a backend (opaque, owned by the plugin) */
typedef struct ii_plugin_kernel ii_plugin_kernel;       /**< a kernel (opaque, owned by the plugin) */
typedef struct ii_plugin_queue ii_plugin_queue;         /**< a queue of a device (opaque, owned by the plugin) */
typedef struct ii_plugin_recording ii_plugin_recording; /**< recorded queue work (opaque, owned by the plugin) */

/** Arguments of one kernel execution (see `ii::backend::KernelContext`). */
typedef struct ii_plugin_run_context {
    const ii_plugin_tensor* inputs;
    int32_t input_count;
    int32_t output_count;
    const ii_plugin_tensor* outputs;
    void* workspace; /**< 64-byte aligned, on the device */
    size_t workspace_size;
    ii_plugin_queue* queue;             /**< a queue of the backend's device; NULL for host backends */
    const ii_plugin_executor* executor; /**< valid during the call; the plugin may copy the struct */
} ii_plugin_run_context;

/** What ii passes to `create_backends`. Valid during the call, except `log` and `log_context`, valid while the
 *  backends live. */
typedef struct ii_plugin_host {
    uint32_t struct_size;
    uint32_t abi_version;
    const char* ii_version;
    void* log_context;
    void (*log)(void* log_context, int32_t level, const char* message);
    void* options_context;
    /** A configuration value (`ContextOptions::plugin_options`, e.g. "cuda.device"), or NULL. "test_devices" holds
     *  the simulated devices requested from ii (`ContextOptions::test_devices` / `II_TEST_DEVICES`). */
    const char* (*get_option)(void* options_context, const char* key);
} ii_plugin_host;

/** Kernel properties (`kernel_flags`). */
enum {
    II_PLUGIN_KERNEL_IN_PLACE = 1,       /**< `Kernel::can_run_in_place(input)` */
    II_PLUGIN_KERNEL_CAPTURES_INPUT = 2, /**< `Kernel::captures_input(input)` */
    II_PLUGIN_KERNEL_RECORDABLE = 4      /**< `Kernel::can_record()`; asked with input -1 */
};

/** The function table of a plugin. */
typedef struct ii_plugin_v1 {
    uint32_t struct_size;   /**< sizeof(ii_plugin_v1) of the plugin */
    uint32_t abi_version;   /**< II_PLUGIN_ABI_VERSION of the plugin */
    const char* name;       /**< e.g. "ii-backend-cuda" */
    const char* version;    /**< version of the plugin */
    const char* ii_version; /**< version of the ii sources the plugin was built from (informational) */

    /**
     * Creates the plugin's backends for one context: at most `capacity` of them, in order of preference. Returns
     * `II_UNAVAILABLE` (with the reason) when the plugin cannot work here — no driver, no device, missing SDK
     * libraries; ii then lists the plugin as unavailable. Creating zero backends is fine.
     */
    int32_t (*create_backends)(const ii_plugin_host* host, ii_plugin_backend** backends, int32_t capacity,
                               int32_t* count, ii_plugin_error* error);
    void (*destroy_backend)(ii_plugin_backend* backend);

    /** Unique backend name, e.g. "ii-cuda". */
    const char* (*backend_name)(const ii_plugin_backend* backend);
    void (*device_info)(const ii_plugin_backend* backend, ii_plugin_device_info* info);

    /** OK if the backend can execute the graph's node; otherwise `II_UNSUPPORTED` and why. */
    int32_t (*supports)(const ii_plugin_backend* backend, const ii_plugin_graph* graph, ii_plugin_error* error);
    /** Estimated time of the node with concrete shapes in microseconds. May be NULL: ii then uses the roofline of
     *  the device info. */
    int32_t (*estimate_us)(const ii_plugin_backend* backend, const ii_plugin_graph* graph,
                           const ii_plugin_shape* inputs, int32_t input_count, const ii_plugin_shape* outputs,
                           int32_t output_count, double* us, ii_plugin_error* error);
    int32_t (*create_kernel)(const ii_plugin_backend* backend, const ii_plugin_graph* graph, ii_plugin_kernel** kernel,
                             ii_plugin_error* error);
    void (*destroy_kernel)(ii_plugin_kernel* kernel);

    /** `II_PLUGIN_KERNEL_*` flags for input `input`; with input -1 the kernel's own flags (`RECORDABLE`). */
    uint32_t (*kernel_flags)(const ii_plugin_kernel* kernel, int32_t input);
    int32_t (*workspace_size)(const ii_plugin_kernel* kernel, const ii_plugin_shape* inputs, int32_t input_count,
                              const ii_plugin_shape* outputs, int32_t output_count, size_t* size,
                              ii_plugin_error* error);
    /** Executes the kernel (on a device: enqueues it on `context->queue`). Must not allocate host memory. */
    int32_t (*run)(const ii_plugin_kernel* kernel, const ii_plugin_run_context* context, ii_plugin_error* error);

    /* Device memory and queues; NULL for backends on the host (device kind 0). */
    int32_t (*allocate)(ii_plugin_backend* backend, size_t size, void** data, ii_plugin_error* error);
    void (*deallocate)(ii_plugin_backend* backend, void* data, size_t size);
    int32_t (*create_queue)(ii_plugin_backend* backend, ii_plugin_queue** queue, ii_plugin_error* error);
    void (*destroy_queue)(ii_plugin_queue* queue);
    int32_t (*copy_to_device)(ii_plugin_queue* queue, void* dst, const void* src, size_t size, ii_plugin_error* error);
    int32_t (*copy_to_host)(ii_plugin_queue* queue, void* dst, const void* src, size_t size, ii_plugin_error* error);
    int32_t (*copy)(ii_plugin_queue* queue, void* dst, const void* src, size_t size, ii_plugin_error* error);
    int32_t (*fill_zero)(ii_plugin_queue* queue, void* dst, size_t size, ii_plugin_error* error);
    int32_t (*synchronize)(ii_plugin_queue* queue, ii_plugin_error* error);

    /* Added later: present when `struct_size` covers them; may be NULL. */

    /** Recording of queue work (`ii::backend::Queue::begin_recording`, CUDA graphs): `begin_recording` returns
     *  `II_UNSUPPORTED` when the queue cannot record; `end_recording` returns the recording (or an error, discarding
     *  it), which `replay` enqueues again and `destroy_recording` frees. */
    int32_t (*begin_recording)(ii_plugin_queue* queue, ii_plugin_error* error);
    int32_t (*end_recording)(ii_plugin_queue* queue, ii_plugin_recording** recording, ii_plugin_error* error);
    int32_t (*replay)(ii_plugin_queue* queue, const ii_plugin_recording* recording, ii_plugin_error* error);
    void (*destroy_recording)(ii_plugin_recording* recording);
} ii_plugin_v1;

/** The entry point a plugin exports. Returns NULL if the plugin does not support `host_abi_version`. */
typedef const ii_plugin_v1* (*ii_backend_plugin_v1_fn)(uint32_t host_abi_version);

/* NOLINTEND(modernize-use-using) */

#ifdef __cplusplus
}
#endif

#endif /* II_PLUGIN_H */
