# Adding an operation

An operation is defined once, by its schema, and then implemented by backends. Adding one touches these places:

| Step | Where |
|------|-------|
| 1. Schema: signature, attributes with defaults, shape/type inference, documentation | `src/ops/*.cpp` |
| 2. Decomposition (composite operations only) | the same file, `decompose = ...` |
| 3. Reference kernel | `src/backends/ref/*.cpp` |
| 4. Builder function (public API) | `include/ii/builder.h`, `src/api/builder.cpp` |
| 5. Shape inference cases | `tests/unit/ops/shape_inference_test.cpp` |
| 6. Conformance cases (NumPy / PyTorch references) | `scripts/gen_op_references.py` → `tests/data/op_references.json` |
| 7. Regenerate the OpSet reference | `ii ops --markdown > docs/opset.md` |

## 1. Schema

```cpp
registry.add(OpSchema{
    .name = "softplus",
    .category = "unary",
    .doc = "log(1 + e^x), computed without overflow for large x.",
    .inputs = {InputSpec{"x"}},
    .outputs = {OutputSpec{"y"}},
    .attrs = {AttrSpec{"beta", AttrKind::kFloat, 1.0}},  // no default value = required attribute
    .infer = infer_float_unary,
    .flags = ir::kOpElementwise,
});
```

Rules:

- **Validate everything in `infer`**: dtypes, ranks, attribute values. A graph that passes `Graph::verify()` must be
  executable; kernels may assume valid inputs. Error messages name the operation and the offending values.
- **Symbolic shapes**: compute output dimensions with `Dim` arithmetic (`+`, `*`, `floor_div`, `ceil_div`,
  `exact_div`). Reject only what is provably wrong; when a property depends on symbol values (e.g. "the dimensions
  are equal"), require structural equality and say so in the error.
- **Attributes are canonical**: nodes always carry every declared attribute (defaults are filled in), so kernels and
  decompositions read them without fallbacks.
- **Strict typing**: operands of arithmetic share one dtype; never insert implicit conversions.
- **Shared helpers** live in `src/ops/infer_util.h` (broadcasting, axes, slices, sliding windows). Kernels use the
  same helpers so that the schema and every backend interpret attributes identically.
- **Cost**: give operations whose work is not proportional to the output size a `cost` function (FLOPs and bytes).

## 2. Decomposition (composites)

A composite operation lists its implementation in simpler operations; every backend without a native kernel gets it
for free. Build the replacement with the sticky-error `ir::Builder` and return the values replacing the outputs:

```cpp
Status decompose_softplus(Builder& b, const Node& node, std::vector<Value*>& outputs) {
    Value* x = node.input(0);
    // max(x, 0) + log(1 + exp(-|x|)) — stable for large |x|
    Value* t = b.op("log", {b.op("add", {b.op("exp", {b.op("neg", {b.op("abs", {x})})}),
                                         b.scalar(1.0, x->dtype())})});
    outputs.push_back(b.op("add", {b.op("relu", {x}), t}));
    return b.status();
}
```

The decomposition must produce exactly the output types of the node (checked by the decomposition pass). Symbolic
shapes work: creation operations (`iota`, `fill`) accept symbolic `shape` attributes.

## 3. Reference kernel

`ii-ref` is the oracle every other backend is compared with, so it favors clarity and accuracy:

- compute floating-point values in double (`load()` / `store<S>()` from `src/backends/ref/common.h` round once);
- integers wrap around (`wrap_add`, `wrap_mul`), as specified in the schema documentation;
- parse attributes when the kernel is created (`Result<std::unique_ptr<Kernel>> factory(const ir::Node&)`), not in
  `run`;
- `run` must not allocate: request scratch memory with `Kernel::workspace_size`;
- absent optional inputs are `TensorView`s with `present() == false` (also when the node has fewer inputs than the
  schema declares).

Register the kernel in the group's `register_*_kernels()`. A backend may decline nodes it cannot handle with a
support check returning `kUnsupported` and a reason — the planner shows it to users.

## 4. Tests

- **Shape inference**: add valid and invalid cases to the table in `shape_inference_test.cpp`; a test fails if an
  operation has no case.
- **Conformance**: add cases to `scripts/gen_op_references.py` with the expected result computed by NumPy or PyTorch in
  float64, including edge cases (NaN, infinities, ties, empty tensors, broadcasting, negative axes), then run
  `python scripts/gen_op_references.py` and commit the updated JSON. `ConformanceDataTest.CoversEveryOperation` fails
  for operations without references. Every case also runs with all composites decomposed.
- **Composites**: add the operation to `kDecompositionCases` in `passes_test.cpp` (native kernel vs. decomposition).
- **Random graphs**: consider adding the operation to `tests/support/random_graph.cpp`, so that pass and backend
  property tests exercise it in combination with others.

## 5. Importers

If ONNX has the operation, map it in `src/formats/onnx/onnx_ops.cpp` (a handler reads the ONNX inputs and attributes
through `NodeContext`, creates ii operations and binds the outputs; parameters ONNX passes as tensors are read with
`ints_input` / `dims_input`, which also work for shape computations tracked symbolically). Then run the ONNX
operator suite (`docs/guides/building.md`, "Model tests") and refresh the list of passing cases with
`ii_onnx_tests <cases> --write-expected tests/onnx/expected_passes.txt`.
