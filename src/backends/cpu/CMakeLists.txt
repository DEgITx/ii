# SPDX-License-Identifier: MIT

# Drivers: shapes, broadcasting, blocking and threading (compiled once, without ISA flags).
ii_add_component(backend_cpu SOURCES
    attention.cpp
    conv.cpp
    cpu_backend.cpp
    elementwise.cpp
    gemm.cpp
    movement.cpp
    reduction.cpp
    scalar_math.cpp)

# Kernels: the same sources compiled once per ISA variant into namespace ii::cpu::<isa>.
foreach(isa IN LISTS II_ISA_VARIANTS)
    ii_add_isa_component(backend_cpu ISA ${isa} SOURCES
        isa/conv.cpp
        isa/elementwise.cpp
        isa/gemm.cpp
        isa/rows.cpp
        isa/table.cpp)
endforeach()
