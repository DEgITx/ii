# SPDX-License-Identifier: MIT

add_subdirectory(core)
add_subdirectory(tensor)
add_subdirectory(ir)
add_subdirectory(ops)
add_subdirectory(backend)
add_subdirectory(plugin)
add_subdirectory(backends)
add_subdirectory(compiler)
add_subdirectory(formats)
add_subdirectory(runtime)
add_subdirectory(llm)
add_subdirectory(api)

ii_finalize_library()
