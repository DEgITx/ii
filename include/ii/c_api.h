/* SPDX-License-Identifier: MIT */
/**
 * @file
 * The stable C interface of ii: for C programs and for bindings to other languages.
 *
 * All objects are opaque handles. Functions return an `ii_status`; on failure `ii_last_error()` returns a message
 * describing the last error on the calling thread. No function throws or aborts on invalid input.
 *
 * @code
 * ii_context* context = NULL;
 * ii_model* model = NULL;
 * ii_program* program = NULL;
 * ii_session* session = NULL;
 * if (ii_context_create(NULL, &context) != II_OK ||
 *     ii_model_from_text(context, text, strlen(text), &model) != II_OK ||
 *     ii_model_compile(model, NULL, &program) != II_OK ||
 *     ii_session_create(program, &session) != II_OK) {
 *     fprintf(stderr, "%s\n", ii_last_error());
 * }
 * @endcode
 *
 * Handles keep the objects they were created from alive: a session may outlive its program, model and context
 * handles. Destroy functions accept NULL.
 */
#ifndef II_C_API_H
#define II_C_API_H

#include <ii/export.h>

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* C declarations: typedef is intended. NOLINTBEGIN(modernize-use-using) */

/** Result of a call. The values match `ii::StatusCode`. */
typedef enum ii_status {
    II_OK = 0,
    II_CANCELLED = 1,
    II_INVALID_ARGUMENT = 2,
    II_NOT_FOUND = 3,
    II_ALREADY_EXISTS = 4,
    II_OUT_OF_RANGE = 5,
    II_OUT_OF_MEMORY = 6,
    II_UNSUPPORTED = 7,
    II_UNAVAILABLE = 8,
    II_IO_ERROR = 9,
    II_DATA_CORRUPTED = 10,
    II_INTERNAL = 11
} ii_status;

/** Element types. The values match `ii::DType`. */
typedef enum ii_dtype {
    II_DTYPE_INVALID = 0,
    II_DTYPE_F32 = 1,
    II_DTYPE_F16 = 2,
    II_DTYPE_BF16 = 3,
    II_DTYPE_F64 = 4,
    II_DTYPE_I8 = 5,
    II_DTYPE_U8 = 6,
    II_DTYPE_I16 = 7,
    II_DTYPE_U16 = 8,
    II_DTYPE_I32 = 9,
    II_DTYPE_U32 = 10,
    II_DTYPE_I64 = 11,
    II_DTYPE_U64 = 12,
    II_DTYPE_BOOL = 13
} ii_dtype;

/** Maximum tensor rank. */
#define II_MAX_RANK 8

/** A dense, row-major host tensor. ii never takes ownership of `data`. */
typedef struct ii_tensor {
    ii_dtype dtype;
    int32_t rank;
    int64_t shape[II_MAX_RANK];
    void* data;
} ii_tensor;

typedef struct ii_context ii_context;
typedef struct ii_model ii_model;
typedef struct ii_program ii_program;
typedef struct ii_session ii_session;

/** Options of `ii_context_create`. Set `struct_size = sizeof(ii_context_options)`. */
typedef struct ii_context_options {
    uint32_t struct_size;
    int32_t num_threads; /**< 0: all logical CPUs */
} ii_context_options;

/** Options of `ii_model_compile`. Set `struct_size = sizeof(ii_compile_options)`. */
typedef struct ii_compile_options {
    uint32_t struct_size;
    const char* backends;   /**< comma-separated allow list in preference order, or NULL for all */
    int32_t fold_constants; /**< non-zero: evaluate constant subgraphs at compile time (default 1) */
} ii_compile_options;

/** The library version, e.g. "0.1.0". */
II_API const char* ii_version(void);

/** The message of the last failed call on this thread ("" if none). Valid until the next failing call. */
II_API const char* ii_last_error(void);

/** Name of a status code, e.g. "InvalidArgument". */
II_API const char* ii_status_name(ii_status status);

II_API ii_status ii_context_create(const ii_context_options* options, ii_context** context);
II_API void ii_context_destroy(ii_context* context);

/** Parses a model in the ii IR text format. `text` need not be NUL-terminated. */
II_API ii_status ii_model_from_text(ii_context* context, const char* text, size_t length, ii_model** model);

/** Loads a model file (ONNX, with external data next to it, or the ii IR text format); `path` is UTF-8. */
II_API ii_status ii_model_load(ii_context* context, const char* path, ii_model** model);

/** Imports a serialized ONNX model from memory (the data is copied). */
II_API ii_status ii_model_from_onnx(ii_context* context, const void* data, size_t size, ii_model** model);

II_API void ii_model_destroy(ii_model* model);

/** Compiles a model. `options` may be NULL. */
II_API ii_status ii_model_compile(const ii_model* model, const ii_compile_options* options, ii_program** program);
II_API void ii_program_destroy(ii_program* program);

/** Number of inputs / outputs of a program and their names (valid while the program lives). */
II_API int32_t ii_program_num_inputs(const ii_program* program);
II_API int32_t ii_program_num_outputs(const ii_program* program);
II_API const char* ii_program_input_name(const ii_program* program, int32_t index);
II_API const char* ii_program_output_name(const ii_program* program, int32_t index);

II_API ii_status ii_session_create(const ii_program* program, ii_session** session);
II_API void ii_session_destroy(ii_session* session);

/** Binds an input without copying: `tensor->data` must stay valid until `ii_session_run` returns. */
II_API ii_status ii_session_set_input(ii_session* session, const char* name, const ii_tensor* tensor);

/** Binds caller memory for an output (optional). */
II_API ii_status ii_session_set_output(ii_session* session, const char* name, const ii_tensor* tensor);

II_API ii_status ii_session_run(ii_session* session);

/** Describes an output of the last run. Unless bound with `ii_session_set_output`, `data` points to session memory
 *  that is valid until the next run. */
II_API ii_status ii_session_get_output(ii_session* session, const char* name, ii_tensor* tensor);

/* NOLINTEND(modernize-use-using) */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* II_C_API_H */
