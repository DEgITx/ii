# ii — дорожная карта

> Этот документ — на русском для удобства планирования. Весь код и остальная документация — на английском.

Принцип: **вертикальные срезы**. Каждый этап заканчивается работающим, протестированным и измеренным результатом,
который можно показать через CLI `ii`. Архитектурные «швы» (Backend API, домены памяти, планировщик, state-значения,
символьные формы, правила градиентов в OpSchema) закладываются с первого этапа, даже если реализация за ними пока
простая, — чтобы позже не переписывать ядро.

**Кроссплатформенность с этапа 0**: Windows, Linux, macOS (tier 1) собираются и проходят все тесты в CI на каждом
коммите. Android подключается на этапе 2 (вместе с NEON-ядрами), затем iOS, WebAssembly и т.д.

Схема зависимостей:

```
0 Фундамент → 1 Ядро (IR + ii-ref) → 2 Быстрый ii-cpu (+Android) → 3 ONNX/CV ─┐
                                                           └→ 4 LLM ─────┼→ 5 Мульти-бэкэнд + CUDA + TensorRT
                                                                         │        ↓
                                                                         └──→ 6 Расширение → 7 Обучение
```

Этапы 3 и 4 технически независимы; ONNX идёт первым по приоритету (первая тестовая модель — YOLOv8).
CI/CD спланирован в [ci.md](ci.md).

---

## Этап 0 — Фундамент (Windows, Linux, macOS)

> **Статус: реализован.** Локально проверен на Windows (MSVC, MSVC shared, MSVC+ASan, clang-cl, MinGW GCC,
> MinGW Clang): 101 тест зелёный. Linux и macOS проверяются в CI. Осталось: первый зелёный прогон CI на GitHub.

**Цель:** скелет проекта, в котором удобно и безопасно разрабатывать всё остальное на всех трёх ОС.

- Репозиторий: git, `.clang-format`, `.clang-tidy`, `.editorconfig`, `.gitignore`, `LICENSE` (MIT), SPDX-заголовки.
- CMake: таргеты по слоям, `CMakePresets.json` (windows-msvc, windows-clang-cl, windows-mingw, linux-gcc,
  linux-clang, macos; режимы debug/release/relwithdebinfo/asan/ubsan/tsan), опции `II_*`, экспорт пакета
  (`find_package(ii)`), хелпер многократной компиляции ISA-специфичных файлов (GCC/Clang/MSVC).
- `core`: `Status`/`Result`/`II_TRY`, логирование с пользовательским приёмником, интерфейс `Allocator` и выровненный
  аллокатор, `SmallVector`, `f16`/`bf16`, `ThreadPool` с `parallel_for`, таймеры.
- `core/platform` (Windows / Linux / macOS): `MappedFile`, `DynLib` (загрузка плагинов), `CpuInfo` (ISA, кэши,
  топология P/E-ядер), привязка потоков, пути кэша.
- Тестовая и бенчмарк-инфраструктура: GoogleTest, Google Benchmark, `ctest`, первый roofline-замер.
- CLI: `ii info` (ОС, CPU, ISA, кэши, ядра) и `ii --version`.
- CI: GitHub Actions по [ci.md](ci.md) — `ci.yml` (lint, матрица tier 1, санитайзеры), `codeql.yml`, Dependabot,
  общий composite action, CMake workflow presets (локально и в CI одни и те же команды).

**Готово, когда:** сборка и тесты зелёные на всех платформах tier 1; тесты `core` (включая исчерпывающие f16/bf16,
платформенный слой на каждой ОС и TSan для пула потоков) проходят; `ii info` корректно описывает машину на каждой ОС;
бенчмарк `parallel_for` и roofline-замер работают.

---

## Этап 1 — Ядро: тензоры, IR, операции, ii-ref, исполнение

> **Статус: реализован.** Локально проверен на Windows (MSVC, MSVC shared, MSVC+ASan, clang-cl, MinGW GCC,
> MinGW Clang). Сделано сверх плана: `ii inspect`, `ii ops` (+ сгенерированный [opset.md](opset.md)),
> JSON-парсер в `core`. Перенесено: деквантизация блочных типов (`q4_0`…) и matmul по ним — этап 4 (вместе с GGUF);
> `if`/`loop` — этап 3; in-place в планировщике памяти, CSE и фьюзинг — этап 2; `abi-interop`, `docs.yml` и
> `nightly.yml` в CI — вместе с первыми загрузчиками форматов (этап 3). По пути найдена и исправлена ошибка этапа 0:
> `std::atomic::wait` в libstdc++ под MinGW терял пробуждения, и пул потоков изредка зависал при завершении.

**Цель:** можно описать модель через builder API, скомпилировать и корректно исполнить на эталонном бэкэнде.

- `Tensor`, `Buffer`, `Device` (host), домены памяти, DType-таблица (включая блочные квант-типы).
- IR: `Graph`/`Node`/`Value`, символьные формы, атрибуты, state-значения, `verify()`, текстовый формат.
- OpSet v1: все примитивы + первые составные (softmax, layer_norm, rms_norm, gelu, silu, linear, sdpa) с
  разложениями; вывод форм; функции стоимости. Поле `gradient` в OpSchema заложено (заполняется на этапе 7).
- Backend API (`Backend`, `Executable`, `KernelBackend`), `ii-ref` со всеми операциями.
- Runtime: `Context`, `Model` (из builder), `Program`, `Session`; простой планировщик (один бэкэнд + фолбэк),
  планировщик памяти, ExecutionPlan.
- Проходы: вывод форм, свёртка констант, DCE, разложение составных.
- Скрипты генерации эталонов (NumPy/PyTorch), харнесс `compare_backends`, генератор случайных графов.
- Публичный C++ API (первая версия) + C ABI (минимум) + пример `examples/mlp`.

**Готово, когда:** тесты соответствия всех операций против NumPy/PyTorch проходят; тесты проходов с численной
эквивалентностью; инвариант «ноль аллокаций в `run()`» проверен; пример MLP исполняется через C++ и C API на всех ОС.

---

## Этап 2 — Быстрый ii-cpu + Android

> **Статус: реализован.** Локально проверен на Windows (MSVC, MSVC shared, MSVC+ASan, clang-cl, MinGW GCC,
> MinGW Clang) и на телефоне Android (Snapdragon 732G, NEON) через `ctest --preset android-arm64-release`.
> AVX-512 проверяется в CI под Intel SDE (на машине разработки AVX-512 нет), Linux ARM64 и macOS — в CI.
> Сделано: SIMD-слой (generic/avx2/avx512/neon) и векторная математика с измеренной точностью; GEMM (BLIS,
> предупаковка константных весов, f16/bf16 веса, GEMV) на уровне OpenBLAS для больших матриц (на всех ядрах — быстрее),
> f32 GEMV — ~80% OpenBLAS, разрыв и план — в [bench/results/stage2-cpu.md](../bench/results/stage2-cpu.md);
> поэлементные операции через интерпретатор программ над тайлами + проход фьюзинга цепочек
> (`fused_elementwise`); редукции, softmax, layer/rms_norm, sdpa (GQA, causal, маски); reshape/permute/concat;
> CSE и шаблон `matmul + bias → linear`; совместное использование памяти (view для reshape, in-place для
> поэлементных); профайлер (`Session::set_profiling`, Chrome trace) и `ii run`; Android (пресеты NDK, запуск тестов
> через adb, эмулятор в CI). Перенесено: ISA-варианты `avx_vnni`/`neon_dotprod`/`neon_i8mm` и `rope` — в этап 4
> (нужны квантованным ядрам и LLM); шаблоны нормализаций/GELU/SiLU из примитивов и conv через GEMM — в этап 3
> (появляются вместе с импортом ONNX); активация в эпилоге GEMM (сейчас — отдельный фьюзнутый проход), автоподбор
> числа потоков для memory-bound GEMV на big.LITTLE и освобождение исходных весов после предупаковки — этап 4.

**Цель:** собственный CPU-бэкэнд, сопоставимый по скорости со зрелыми библиотеками, на любом CPU и ОС.

- SIMD-слой, runtime-диспетчер ISA (`generic`, `avx2`, `avx_vnni`, `avx512`, `neon`, `neon_dotprod`, `neon_i8mm`),
  `II_CPU_ISA`.
- GEMM f32 (BLIS-подход, предупаковка весов), f16/bf16 веса; GEMV.
- Поэлементные операции и редукции с фьюзингом цепочек; softmax, нормализации, rope.
- Распараллеливание с динамической раздачей (гибридные CPU: Intel P/E, Apple, big.LITTLE).
- Планировщик: `ii-cpu` предпочтительнее, `ii-ref` — фолбэк для того, что ещё не ускорено.
- Проходы: фьюзинг шаблонов (linear+act, нормализации), упаковка весов.
- Профайлер + Chrome trace.
- **Android**: пресет NDK (arm64-v8a, x86_64), CI-сборка, тесты на эмуляторе/устройстве.

**Готово, когда:** все тесты операций проходят для `ii-cpu` на всех ISA (AVX-512 — через SDE, NEON — на ARM64 Linux,
macOS и Android); цели микробенчмарков GEMM/поэлементных достигнуты или задокументирован разрыв и план; сравнение с
OpenBLAS — в `bench/results/`.

---

## Этап 3 — ONNX и «классические» модели (первый «настоящий» сценарий)

> **Статус: реализован.** Локально проверен на Windows (MSVC, MSVC shared, MSVC+ASan, clang-cl, MinGW GCC,
> MinGW Clang) и на телефоне Android; Linux ARM64, macOS и эмулятор — в CI. Сделано: собственный декодер protobuf и
> импорт ONNX (~130 операторов, opset 1–23+): символьное отслеживание вычислений над формами (Shape → Gather → Concat →
> Reshape и т.п.), поэтому модели с динамическими batch/sequence импортируются с символьными размерностями; функции
> модели, `If` с известным при загрузке условием, внешние данные (только внутри каталога модели),
> `LoadOptions::dims` / `constant_inputs`. Новые операции `resize`, `topk`. В ii-cpu: свёртки через GEMM с неявным
> im2col (панели упаковываются один раз на изображение) и предупаковкой весов, depthwise-свёртки, пулинги, resize;
> эпилог свёртки — внутренняя операция `fused_conv` (активации, SiLU, residual; batch_norm после импорта становится
> поканальными константами эпилога). `.npy` / `.npz` (свой inflate) в публичном API `ii/io.h`; CLI: `ii run` с
> `.npz`/`.npy`/изображениями, `--save-outputs`, `--top`, `ii bench`, `ii inspect` для ONNX; пример `examples/yolo`
> (letterbox, декодирование, NMS, отрисовка). Тесты: официальный набор операторных тестов ONNX — 1005 из 1884 случаев
> проходят на ii-ref и ii-cpu (CI `onnx-conformance` со списком ожидаемых), модели YOLOv8n, ResNet-50 (2 экспорта),
> MobileNetV2 (2 экспорта), BERT-base (2 экспорта) совпадают с ONNX Runtime (~1e-6; CI `models` и `nightly.yml`),
> фаззеры ONNX и npy/npz (+ 30 минут каждого в nightly). Задержка: ×1.1–1.25 к ONNX Runtime на BERT и YOLOv8n,
> ×1.5 на ResNet-50, ×1.9 на MobileNetV2 — разрыв и план в
> [bench/results/stage3-onnx.md](../bench/results/stage3-onnx.md) (раскладка NCHWc и прямые ядра свёрток).
> Перенесено: NCHWc / прямые ядра / Winograd — следующий шаг оптимизации CPU; `if`/`loop` с условием во время
> выполнения и `nms` как операция — вместе с выходами, размер которых зависит от данных (ONNX NonMaxSuppression,
> NonZero); распознавание attention-шаблонов в `sdpa` — этап 4 (вместе с attention для LLM); conv_transpose, 3-D
> окна и cubic resize в ii-cpu — по необходимости (сейчас ii-ref); `release.yml` (v0.1), `abi-interop`, `docs.yml`.

**Цель:** запуск произвольных ONNX-моделей (CV, энкодеры, аудио); первая цель — YOLOv8.

- Собственный декодер protobuf, импорт ONNX (opset 13–21+), external data, таблица маппинга операций; фаззинг.
- Свёртки (im2col+GEMM → прямые / Winograd / depthwise), пулинги, фолдинг batch_norm, раскладка NCHWc.
- Операции, нужные детекторам: resize/upsample, sigmoid/silu-цепочки, concat/split, `topk`, `nms` (как составная
  операция с эталоном).
- `if`/`loop` (подграфы), распознавание attention-шаблонов в `sdpa`.
- CLI: `ii run` с `.npy/.npz`/изображениями, `ii inspect`, `ii bench`.
- Пример `examples/yolo`: загрузка изображения, инференс, декодирование боксов + NMS, отрисовка результата.
- Тесты моделей в CI: манифест моделей (URL + sha256), кэш, сверка с ONNX Runtime.

**Готово, когда:** YOLOv8n/s, ResNet-50, MobileNetV2, BERT-base дают результаты в допуске относительно ONNX Runtime
(для YOLO — совпадение боксов/классов после NMS); задержка сопоставима с ORT CPU EP; ONNX-фаззер работает в CI.

---

## Этап 4 — LLM

> **Статус: реализован.** Локально проверен на Windows (MinGW GCC/Clang, MSVC, MSVC shared, MSVC+ASan, clang-cl)
> и на телефоне Android (Snapdragon 732G); Linux, macOS и эмулятор — в CI. Сделано: кодеки всех блочных типов GGUF
> (`q4_0` … `q8_K`, побайтно совместимы с gguf-py / llama.cpp), чтение и запись GGUF (`ii/gguf.h`, тензоры —
> представления mmap), safetensors (в т.ч. шардированные), библиотека `ii::llm` (`ii/llm.h`: `Llm`, `Sequence`,
> `Chat`, `Tokenizer`, `Sampler`) с архитектурами llama, qwen2, qwen3, gemma, gemma2, gemma3, phi3 из GGUF или
> каталога Hugging Face; новые операции `rope`, `slice_update`, у `sdpa` — `kv_len` (предвыделенный KV-кэш),
> скользящее окно, `softcap`, 16-битные K/V; веса f16/bf16 и блочные кванты в `matmul`/`linear`; KV-кэш — state,
> который `slice_update` пишет на месте (runtime не копирует), кэш планов для чередования prefill/decode;
> токенизаторы SentencePiece и byte-level BPE (пре-токенизаторы GPT-2 / Llama 3 / Qwen 2 на таблицах Unicode),
> интерпретатор Jinja для шаблонов чата, сэмплеры. В ii-cpu: квантованные произведения с 8-битными активациями
> (как в llama.cpp), панели по 8 строк для q4_0/q5_0/q8_0/q4_K/q6_K (перепаковка при компиляции), новые
> ISA-варианты `avx_vnni` и `neon_dotprod`, `sdpa` с кэшем, `rope`. CLI: `ii generate`, `ii chat`,
> `ii bench --llm` (pp/tg), `ii perplexity`, `ii quantize` (в т.ч. смеси q4_k_m/q5_k_m), `ii tokenize`,
> `ii inspect` для GGUF. Тесты: крошечные случайные модели шести архитектур в обоих форматах совпадают с
> transformers по логитам всех позиций (в т.ч. пошагово через KV-кэш), 66 рендеров 11 реальных шаблонов чата —
> байт в байт как в transformers, фаззеры GGUF/safetensors и токенизаторов/Jinja, точные тесты квантованных ядер
> каждого ISA. Реальные модели (`scripts/llm_tests.py`, nightly): TinyLlama-1.1B и Qwen2.5-0.5B — логиты
> ~4e-6 от transformers, жадная генерация совпадает на 128 / 64 токенах, перплексия wikitext-2 совпадает до 4 знаков
> (9.1393 против 9.1365 на 8 фрагментах; 9.2903 = 9.2903 на 20), q4_k_m — +3.9%. Скорость (Core Ultra 7 265KF,
> 20 потоков): Qwen2.5-0.5B q4_k_m pp512 ≈ 2570 / tg128 ≈ 157 ток/с, TinyLlama q4_k_m ≈ 1170 / 99, Llama-3.2-1B
> q4_k_m ≈ 1180 / 85; телефон: Qwen2.5-0.5B pp128 134 / tg64 21 ток/с —
> [bench/results/stage4-llm.md](../bench/results/stage4-llm.md). Против llama.cpp b12728 на той же машине и тех же
> GGUF: pp512 у ii быстрее на всех пяти моделях (×1.08–2.25), tg128 — 0.94–0.98 от llama.cpp (оба упираются в
> память; план — prefetch панелей). Перенесено: tiled flash-attention для длинных промптов, панели для q5_K и
> типов q2_K/q3_K, `avx512_vnni`/`neon_i8mm`, yarn/longrope, MoE, пакетная обработка последовательностей,
> конвертация HF → GGUF в `ii quantize`.

**Цель:** `ii chat model.gguf` быстро работает на обычном CPU (в т.ч. на Android).

- Загрузчики: GGUF (mmap, zero-copy квант-тензоры), safetensors + HF `config.json`; фаззинг.
- Квант-типы и ядра: `q8_0`, `q4_0`, `q4_k`, `q5_k`, `q6_k` — эталон в `ii-ref`, быстрые GEMV/GEMM в `ii-cpu`
  (AVX2, AVX-VNNI, NEON dotprod/i8mm).
- `sdpa` (flash-attention для CPU), KV-cache как state, символ `seq`.
- Архитектуры: llama (Llama 2/3, Mistral, TinyLlama), qwen2/2.5/3, gemma, phi.
- Токенизаторы (byte-level BPE, SentencePiece), шаблоны чата (подмножество Jinja), сэмплеры, стриминг.
- CLI: `ii generate`, `ii chat`, `ii bench --llm`, `ii quantize`.

**Готово, когда:** логиты tiny-llama совпадают с transformers; на реальной модели greedy-генерация совпадает с
эталоном на N токенов, перплексия в допуске; `tg128`/`pp512` в пределах 10% от llama.cpp на той же машине и
квантизации (или задокументирован разрыв и план).

---

## Этап 5 — Мульти-бэкэнд: ii-cuda + TensorRT

**Цель:** библиотека сама выбирает между CPU, собственными CUDA-ядрами и TensorRT и делит модель между устройствами.

- Полноценный планировщик: стоимостная модель, разбиение графа, домены памяти, пересылки, события, ограничения
  памяти, автоматический offload слоёв, автотюнинг с кэшем, `ii inspect --plan` с объяснениями.
- Тестовый бэкэнд `ii-test-remote` (межустройственные сценарии без GPU).
- Плагинный ABI `ii_backend_plugin_v1`, загрузка плагинов; плагин без рантайма SDK сообщает «недоступен».
- **`ii-cuda`**: свои ядра (квантованный matmul, attention, фьюженные поэлементные через NVRTC), cuBLASLt,
  CUDA Graphs для decode; LLM на GPU с offload части слоёв на CPU.
- **Плагин TensorRT**: трансляция партиций IR в `INetworkDefinition`, профили динамических форм, кэш движков на
  диске, общий домен памяти с `ii-cuda` (смешанные планы TRT + свои ядра без копий через host).
- Self-hosted GPU-раннер в CI (машина с RTX 5080).
- `ii check-backends`.

**Готово, когда:** дифференциальные тесты проходят для `ii-cuda` и TensorRT; модель, не помещающаяся в VRAM,
автоматически делится между GPU и CPU; на наборе моделей (ResNet-50, BERT, LLM) планировщик выбирает план не хуже
лучшего ручного; производительность сравнена с `trtexec` и llama.cpp CUDA.

---

## Этап 6 — Расширение

По приоритетам, определяемым бенчмарками и запросами:

- платформы: iOS, WebAssembly (`wasm_simd128`), Windows ARM64, Linux RISC-V (`rvv`);
- бэкэнды: `ii-vulkan` (универсальный GPU, включая Android), `ii-metal` (macOS/iOS); плагины ONNX Runtime
  (длинный хвост ONNX), OpenVINO (включая Intel NPU), CoreML (ANE); oneDNN (AMX) / XNNPACK / KleidiAI — там,
  где бенчмарки показывают выигрыш;
- нативный формат `.ii` с предупакованными весами, `ii convert`;
- LLM: MoE, paged KV-cache и continuous batching, speculative decoding, грамматики, мультимодальность;
- серверный режим (`ii serve`, OpenAI-совместимый API);
- привязки: Python (nanobind), C#, Rust — поверх C ABI;
- минимальная сборка под конкретные модели (`II_MINIMAL`).

---

## Этап 7 — Обучение

**Цель:** обучение и дообучение моделей средствами ii на любом бэкэнде.

- Правила градиентов для всех операций, проход autodiff, фьюженные backward-операции (в т.ч. для `ii-cuda`).
- `nn::Module`, трассировка в граф, функции потерь, оптимизаторы (SGD, Adam, AdamW) как фьюженные операции.
- Шаг обучения как один `Program`, DataLoader, чекпоинты (safetensors).
- Mixed precision, gradient checkpointing, LoRA/QLoRA для LLM.
- CLI: `ii train`.

**Готово, когда:** проверка градиентов конечными разностями для всех операций; MNIST-MLP и маленький CNN достигают
эталонной точности; кривая обучения совпадает с PyTorch при одинаковой инициализации; LoRA-дообучение маленькой LLM
снижает loss на целевом наборе (на CPU и на CUDA).

---

## Окружение разработки (на момент планирования)

| Компонент | Значение | Следствие |
|-----------|----------|-----------|
| ОС | Windows 11 Pro | основная машина разработки; Linux и macOS — через CI (и WSL2 для Linux локально) |
| CPU | Intel Core Ultra 7 265KF: 20 ядер (8P + 12E), AVX2, FMA, F16C, AVX-VNNI, **без AVX-512** | первые ядра — `avx2` + `avx_vnni`; гибридные ядра → динамическая раздача; AVX-512/AMX — через Intel SDE |
| GPU | NVIDIA GeForce RTX 5080 (Blackwell, sm_120) | цель `ii-cuda` и TensorRT; self-hosted GPU-раннер |
| Компиляторы | MSVC 14.44 (VS 2022 Build Tools), MinGW-w64 GCC 14.2, Clang 19.1 | все три — в пресетах; MSVC — host-компилятор для CUDA. `vswhere` отсутствует в стандартном месте — CMake-пресеты должны находить MSVC через окружение `vcvars` |
| Сборка | CMake 4.3, Ninja 1.12 | ок |
| CUDA | 13.2 (`C:\CUDA`) | ок |
| TensorRT | 10.16.1 (`C:\TensorRT`) | плагин этапа 5 |
| Vulkan SDK | 1.4.357 (`C:\VulkanSDK\1.4.357.0`) | `ii-vulkan` на этапе 6 |
| cuDNN | не найден | не нужен (TensorRT 10 и cuBLASLt обходятся без него) |
| Python | 3.13 | генерация эталонов и сравнение бенчмарков |
