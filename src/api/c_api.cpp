// SPDX-License-Identifier: MIT
#include <ii/c_api.h>
#include <ii/model.h>
#include <ii/version.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <new>
#include <string>
#include <vector>

struct ii_context {
    ii::Context context;
};
struct ii_model {
    ii::Model model;
};
struct ii_program {
    ii::Program program;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
};
struct ii_session {
    ii::Session session;
};

// The C enumerations mirror the C++ ones.
static_assert(II_INTERNAL == static_cast<int>(ii::StatusCode::kInternal));
static_assert(II_DATA_CORRUPTED == static_cast<int>(ii::StatusCode::kDataCorrupted));
static_assert(II_DTYPE_F32 == static_cast<int>(ii::DType::kF32));
static_assert(II_DTYPE_BF16 == static_cast<int>(ii::DType::kBF16));
static_assert(II_DTYPE_I64 == static_cast<int>(ii::DType::kI64));
static_assert(II_DTYPE_BOOL == static_cast<int>(ii::DType::kBool));
static_assert(II_MAX_RANK == ii::kMaxRank);

namespace {

thread_local std::string g_last_error;

ii_status fail(const ii::Status& status) {
    g_last_error = status.to_string();
    return static_cast<ii_status>(status.code());
}

ii_status fail(ii_status code, const char* message) {
    g_last_error = message;
    return code;
}

/// Runs `body` and converts exceptions (only std::bad_alloc is expected) into status codes: nothing may escape
/// through the C boundary.
template <class Body>
ii_status guarded(Body&& body) noexcept {
    try {
        return body();
    } catch (const std::bad_alloc&) {
        return fail(II_OUT_OF_MEMORY, "out of memory");
    } catch (...) {
        return fail(II_INTERNAL, "unexpected exception");
    }
}

static_assert(sizeof(ii_dtype) == sizeof(std::int32_t));

ii::Result<ii::Tensor> to_tensor(const ii_tensor* tensor) {
    if (tensor == nullptr || tensor->rank < 0 || tensor->rank > II_MAX_RANK || tensor->data == nullptr) {
        return ii::errors::invalid_argument("invalid ii_tensor");
    }
    // Read as an integer: a C caller may store any int in the enum, and loading a value outside the enum's range is
    // undefined in C++.
    std::int32_t dtype = 0;
    std::memcpy(&dtype, &tensor->dtype, sizeof(dtype));
    if (dtype <= II_DTYPE_INVALID || dtype > II_DTYPE_BOOL) {
        return ii::errors::invalid_argument("invalid ii_tensor dtype {}", dtype);
    }
    ii::Shape shape;
    for (int32_t i = 0; i < tensor->rank; ++i) {
        if (tensor->shape[i] < 0) {
            return ii::errors::invalid_argument("negative dimension in ii_tensor");
        }
        shape.push_back(tensor->shape[i]);
    }
    return ii::Tensor::wrap(tensor->data, static_cast<ii::DType>(dtype), shape);
}

}  // namespace

extern "C" {

const char* ii_version(void) {
    return ii::build_info().version.data();
}

const char* ii_last_error(void) {
    return g_last_error.c_str();
}

const char* ii_status_name(ii_status status) {
    // StatusCode names are string literals, hence NUL-terminated.
    return ii::to_string(static_cast<ii::StatusCode>(status)).data();
}

ii_status ii_context_create(const ii_context_options* options, ii_context** context) {
    return guarded([&]() -> ii_status {
        if (context == nullptr) {
            return fail(II_INVALID_ARGUMENT, "context must not be NULL");
        }
        *context = nullptr;
        ii::ContextOptions cpp_options;
        if (options != nullptr) {
            if (options->struct_size < sizeof(ii_context_options)) {
                return fail(II_INVALID_ARGUMENT, "ii_context_options.struct_size is too small");
            }
            cpp_options.num_threads = options->num_threads;
        }
        ii::Result<ii::Context> created = ii::Context::create(std::move(cpp_options));
        if (!created.ok()) {
            return fail(created.status());
        }
        *context = new ii_context{std::move(*created)};
        return II_OK;
    });
}

void ii_context_destroy(ii_context* context) {
    delete context;
}

ii_status ii_model_from_text(ii_context* context, const char* text, size_t length, ii_model** model) {
    return guarded([&]() -> ii_status {
        if (context == nullptr || model == nullptr || (text == nullptr && length > 0)) {
            return fail(II_INVALID_ARGUMENT, "invalid argument to ii_model_from_text");
        }
        *model = nullptr;
        ii::Result<ii::Model> parsed = ii::Model::from_text(context->context, std::string_view(text, length));
        if (!parsed.ok()) {
            return fail(parsed.status());
        }
        *model = new ii_model{std::move(*parsed)};
        return II_OK;
    });
}

ii_status ii_model_load(ii_context* context, const char* path, ii_model** model) {
    return guarded([&]() -> ii_status {
        if (context == nullptr || model == nullptr || path == nullptr) {
            return fail(II_INVALID_ARGUMENT, "invalid argument to ii_model_load");
        }
        *model = nullptr;
        const std::string_view text(path);
        ii::Result<ii::Model> loaded =
            ii::Model::load(context->context, std::filesystem::path(std::u8string(text.begin(), text.end())));
        if (!loaded.ok()) {
            return fail(loaded.status());
        }
        *model = new ii_model{std::move(*loaded)};
        return II_OK;
    });
}

ii_status ii_model_from_onnx(ii_context* context, const void* data, size_t size, ii_model** model) {
    return guarded([&]() -> ii_status {
        if (context == nullptr || model == nullptr || (data == nullptr && size > 0)) {
            return fail(II_INVALID_ARGUMENT, "invalid argument to ii_model_from_onnx");
        }
        *model = nullptr;
        ii::Result<ii::Model> imported = ii::Model::from_onnx(
            context->context, std::span<const std::byte>(static_cast<const std::byte*>(data), size));
        if (!imported.ok()) {
            return fail(imported.status());
        }
        *model = new ii_model{std::move(*imported)};
        return II_OK;
    });
}

void ii_model_destroy(ii_model* model) {
    delete model;
}

ii_status ii_model_compile(const ii_model* model, const ii_compile_options* options, ii_program** program) {
    return guarded([&]() -> ii_status {
        if (model == nullptr || program == nullptr) {
            return fail(II_INVALID_ARGUMENT, "invalid argument to ii_model_compile");
        }
        *program = nullptr;
        ii::CompileOptions cpp_options;
        if (options != nullptr) {
            if (options->struct_size < sizeof(ii_compile_options)) {
                return fail(II_INVALID_ARGUMENT, "ii_compile_options.struct_size is too small");
            }
            cpp_options.fold_constants = options->fold_constants != 0;
            if (options->backends != nullptr) {
                std::string_view list = options->backends;
                while (!list.empty()) {
                    const std::size_t comma = list.find(',');
                    const std::string_view name = list.substr(0, comma);
                    if (!name.empty()) {
                        cpp_options.backends.emplace_back(name);
                    }
                    list = comma == std::string_view::npos ? std::string_view() : list.substr(comma + 1);
                }
            }
        }
        ii::Result<ii::Program> compiled = model->model.compile(cpp_options);
        if (!compiled.ok()) {
            return fail(compiled.status());
        }
        auto* handle = new ii_program{std::move(*compiled), {}, {}};
        for (const ii::TensorInfo& info : handle->program.inputs()) {
            handle->input_names.push_back(info.name);
        }
        for (const ii::TensorInfo& info : handle->program.outputs()) {
            handle->output_names.push_back(info.name);
        }
        *program = handle;
        return II_OK;
    });
}

void ii_program_destroy(ii_program* program) {
    delete program;
}

int32_t ii_program_num_inputs(const ii_program* program) {
    return program == nullptr ? 0 : static_cast<int32_t>(program->input_names.size());
}

int32_t ii_program_num_outputs(const ii_program* program) {
    return program == nullptr ? 0 : static_cast<int32_t>(program->output_names.size());
}

const char* ii_program_input_name(const ii_program* program, int32_t index) {
    if (program == nullptr || index < 0 || static_cast<std::size_t>(index) >= program->input_names.size()) {
        return nullptr;
    }
    return program->input_names[static_cast<std::size_t>(index)].c_str();
}

const char* ii_program_output_name(const ii_program* program, int32_t index) {
    if (program == nullptr || index < 0 || static_cast<std::size_t>(index) >= program->output_names.size()) {
        return nullptr;
    }
    return program->output_names[static_cast<std::size_t>(index)].c_str();
}

ii_status ii_session_create(const ii_program* program, ii_session** session) {
    return guarded([&]() -> ii_status {
        if (program == nullptr || session == nullptr) {
            return fail(II_INVALID_ARGUMENT, "invalid argument to ii_session_create");
        }
        *session = nullptr;
        ii::Result<ii::Session> created = program->program.new_session();
        if (!created.ok()) {
            return fail(created.status());
        }
        *session = new ii_session{std::move(*created)};
        return II_OK;
    });
}

void ii_session_destroy(ii_session* session) {
    delete session;
}

ii_status ii_session_set_input(ii_session* session, const char* name, const ii_tensor* tensor) {
    return guarded([&]() -> ii_status {
        if (session == nullptr || name == nullptr) {
            return fail(II_INVALID_ARGUMENT, "invalid argument to ii_session_set_input");
        }
        ii::Result<ii::Tensor> wrapped = to_tensor(tensor);
        if (!wrapped.ok()) {
            return fail(wrapped.status());
        }
        const ii::Status status = session->session.set_input(name, *wrapped);
        return status.ok() ? II_OK : fail(status);
    });
}

ii_status ii_session_set_output(ii_session* session, const char* name, const ii_tensor* tensor) {
    return guarded([&]() -> ii_status {
        if (session == nullptr || name == nullptr) {
            return fail(II_INVALID_ARGUMENT, "invalid argument to ii_session_set_output");
        }
        ii::Result<ii::Tensor> wrapped = to_tensor(tensor);
        if (!wrapped.ok()) {
            return fail(wrapped.status());
        }
        const ii::Status status = session->session.set_output(name, *wrapped);
        return status.ok() ? II_OK : fail(status);
    });
}

ii_status ii_session_run(ii_session* session) {
    return guarded([&]() -> ii_status {
        if (session == nullptr) {
            return fail(II_INVALID_ARGUMENT, "session must not be NULL");
        }
        const ii::Status status = session->session.run();
        return status.ok() ? II_OK : fail(status);
    });
}

ii_status ii_session_get_output(ii_session* session, const char* name, ii_tensor* tensor) {
    return guarded([&]() -> ii_status {
        if (session == nullptr || name == nullptr || tensor == nullptr) {
            return fail(II_INVALID_ARGUMENT, "invalid argument to ii_session_get_output");
        }
        ii::Result<ii::Tensor> output = session->session.output(name);
        if (!output.ok()) {
            return fail(output.status());
        }
        if (!output->is_contiguous() || output->dtype() > ii::DType::kBool) {
            return fail(II_UNSUPPORTED, "the output cannot be described as a dense ii_tensor");
        }
        tensor->dtype = static_cast<ii_dtype>(output->dtype());
        tensor->rank = output->rank();
        for (int i = 0; i < II_MAX_RANK; ++i) {
            tensor->shape[i] = i < output->rank() ? output->shape()[i] : 0;
        }
        tensor->data = output->data();
        return II_OK;
    });
}

}  // extern "C"
