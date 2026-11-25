/* SPDX-License-Identifier: MIT */
/* Runs a small model through the C interface. The model is written in the ii IR text format:
 *
 *   y = relu(x @ w + b)      x: [N, 3], w: [3, 2], b: [2]
 */
#include <ii/c_api.h>

#include <stdio.h>
#include <string.h>

static const char* kModel = "graph mlp {\n"
                            "  input %x : f32[N, 3]\n"
                            "  const %w : f32[3, 2] = f32[3, 2] {1, -1, 2, 0.5, -3, 1}\n"
                            "  const %b : f32[2] = f32[2] {0.5, -0.25}\n"
                            "  %h : f32[N, 2] = matmul(%x, %w)\n"
                            "  %z : f32[N, 2] = add(%h, %b)\n"
                            "  %y : f32[N, 2] = relu(%z)\n"
                            "  output y = %y\n"
                            "}\n";

static int check(ii_status status) {
    if (status != II_OK) {
        fprintf(stderr, "error (%s): %s\n", ii_status_name(status), ii_last_error());
        return 0;
    }
    return 1;
}

int main(void) {
    ii_context* context = NULL;
    ii_model* model = NULL;
    ii_program* program = NULL;
    ii_session* session = NULL;
    int ok = 0;
    float input[2 * 3] = {1, 2, 3, -1, 0, 1};
    ii_tensor x;
    ii_tensor y;
    int64_t i;

    printf("ii %s\n", ii_version());
    memset(&x, 0, sizeof x);
    x.dtype = II_DTYPE_F32;
    x.rank = 2;
    x.shape[0] = 2;
    x.shape[1] = 3;
    x.data = input;

    if (check(ii_context_create(NULL, &context)) &&
        check(ii_model_from_text(context, kModel, strlen(kModel), &model)) &&
        check(ii_model_compile(model, NULL, &program)) && check(ii_session_create(program, &session)) &&
        check(ii_session_set_input(session, "x", &x)) && check(ii_session_run(session)) &&
        check(ii_session_get_output(session, "y", &y))) {
        for (i = 0; i < y.shape[0] * y.shape[1]; ++i) {
            printf("y[%lld] = %g\n", (long long)i, ((const float*)y.data)[i]);
        }
        ok = 1;
    }

    ii_session_destroy(session);
    ii_program_destroy(program);
    ii_model_destroy(model);
    ii_context_destroy(context);
    return ok ? 0 : 1;
}
