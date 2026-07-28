# Разбор скорости обучения

Документ связывает исходное наблюдение issue #7 с измерениями RX 9070 XT из
issue #13. Здесь optimizer step, token budget и wall-clock считаются отдельно:
уменьшить число micro-batches в шаге — не ускорение, если вместе с ним незаметно
уменьшилось число обучающих токенов.

## Что означало `40 / 60000`

Burn TUI считал optimizer steps. При `micro_batch=1`, `accum=96` и
`seq_len=2048` один шаг содержит 196 608 токенов. Измеренные 1 700 tok/s дают
`1700 × 3600 / 196608 = 31.1` шага в час, поэтому около 40 первых шагов — не
зависание и не ошибка progress bar.

Старые 60 000 шагов означали 11.80B токенов с этой формой batch и около 80.3
суток при постоянной скорости. Default сокращён до 12 500 шагов: для `tiny`
это 3.2768B токенов (`8 × 16 × 2048`), близко к 20 токенам на параметр. Startup
теперь печатает steps, tokens/step и полный token budget до первой аллокации.

## Найденные причины

1. Старый schedule планировал почти в пять раз больше compute-efficient token
   budget.
2. Logging читал scalar loss после каждого micro-batch и каждый раз
   синхронизировал host/device. Теперь loss суммируется на device и читается
   один раз за окно.
3. Burn/burn-mamba в закреплённых ревизиях не имеют готового AMP. Глобальный
   bf16/f16 остаётся небезопасной заменой autocast + GradScaler; измеренный
   production path использует fp16 только для output-head, трёх FFN GEMM и
   Mamba input/output projections, с fp32 master/norm/residual/SSD
   coefficients/state/logits/loss и собственным dynamic loss scaler.
4. Внешний Burn checkpointing повторял целый block, а внутренний
   burn-mamba `SerialRecalculated` повторял SSD ещё раз.
5. CubeCL удерживал до 128 Fusion streams и вместе с ними live buffers. Один
   stream оказался и быстрее, и стабильнее по VRAM.
6. 512-wide модель запускала много узких GEMM и 20 последовательных слоёв.
   Форма 640 × 12 сохранила параметр/FLOP budget, но лучше загрузила GPU.

Muon не объясняет разницу порядков величины: ортогонализация вызывается один раз
после всех accumulated forward/backward. `--muon false` добавляет второй moment
AdamW и расходует больше памяти; без отдельного A/B это не shortcut.

## Измерения RX 9070 XT

Все варианты — настоящий synchronized optimizer step с Muon, fp32 и полным
language-model loss. Короткие эксперименты выполнялись в GitHub Actions, а
полная методика и ссылки на runs находятся в [`KERNELS.md`](KERNELS.md).

| изменение | matched baseline | candidate | эффект |
| --- | ---: | ---: | ---: |
| SSD `recalculated → serial`, ROCm | 3 616 | 4 485 tok/s | +24.0% |
| SSD `recalculated → serial`, Vulkan | 4 834 | 6 274 tok/s | +29.8% |
| checkpoint on, 6×8 → off, 4×12 | 7 359 | 8 406 tok/s | +14.2% |
| CubeCL streams 128 → 1 | 8 406 | 8 608 tok/s | +2.4% |
| shape 512×20 → 640×12 | 8 709 | 9 457 tok/s | +8.6% |
| tensor K4 → CubeCL K4 | 9 571 | 9 818 tok/s | +2.6% |
| CubeCL K4 → fused rank-one scan | 9 818 | 10 368 tok/s | +5.6% |

До precision-работы итоговая production-форма с fused scan дала **10 578
tok/s** при пике 14.074 GiB. `tiny-turbo` поэтому выбирает 640 × 12,
micro-batch 4, `serial`, checkpointing off и один CubeCL stream. Accumulation
по умолчанию поднят до 32: effective batch остаётся
`4 × 32 × 1024 = 131072`, то есть скорость не куплена сокращением обучения.

### P0 freeze для precision-работы

[Run 30310985422](https://github.com/uselessgoddess/quasar/actions/runs/30310985422)
зафиксировал новый reference на source commit `fc0e5f8746905bb931fcc1be21eb9dd849d6d05f`
с тем же production recipe:

| samples | median | min/max | throughput | effective | peak VRAM |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 9 | 12.415 s | 12.374–16.706 s | **10 558 tok/s** | **5.11 TFLOP/s** | **14.516 GiB** |

Первое measured значение попало в дополнительный autotune и дало разброс 35%;
по обязательному правилу окно автоматически расширилось с 3 до 9. Значения
2–9 лежат в диапазоне 10 540–10 593 tok/s, поэтому медиана не определяется
этим выбросом. Карта доходила до 100% busy, 3.16 GHz, 357 W и junction 84 °C,
то есть downclock/throttling baseline не объясняет.

Profiler после warm-up насчитал **6 668 launches** за micro-batch. Их GPU
timestamps составили 392.10 ms, тогда как wall time стабильного sample —
1.040 s: около 62% wall time остаётся вне записанной kernel duration. Пять
семейств matmul дали 1 221 launch и 264.35 ms, то есть 67.4% GPU timestamps.
Текущий bottleneck поэтому смешанный: матричный compute плюс
launch/orchestration overhead. Изолированный roofline и precision gate
записаны в [`ROOFLINE.md`](ROOFLINE.md).

Увеличивать micro-batch выше измеренного нельзя по принципу «раз помещается»:
6×8 без checkpointing не завершился OOM, но упал до 2 715 tok/s из-за memory
pressure. Пик VRAM и steady throughput нужно проверять вместе.

### P2: bf16 только для tied head — отклонён

[Run 30312309297](https://github.com/uselessgoddess/quasar/actions/runs/30312309297)
проверил минимальный precision seam на commit
`49185deb6657c4bb23b08bb6aad6ddbd3ea54179`: параметры, optimizer state,
hidden states, logits, softmax и loss оставались fp32; в bf16 переводились
только вход и связанный embedding weight на время `hidden × weightᵀ`. Обе
стороны получили одинаковые seed, синтетические токены, 131 072 токена/step,
один warm-up и девять measured steps. Первый measured step снова содержал
autotune, поэтому в таблице медиана всех девяти и полный min/max:

| вариант | median | min/max | tok/s | effective | peak VRAM | решение |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fp32 control | 12.470 s | 12.448–14.488 s | 10 511 | 5.08 TFLOP/s | 14.117 GiB | reference |
| bf16 tied head | 10.852 s | 10.837–12.846 s | **12 078** | **5.84 TFLOP/s** | **13.359 GiB** | **revert** |

Скорость выросла на 14.9%, FLOPs/token не менялись, а память уменьшилась на
0.758 GiB. Начальная диагностика тоже выглядела приемлемо: activation
`[-4.8125, 5.0]`, mean `3.240e-3`, grad norm `1.216` против `1.228`, loss
scale 1 и ноль NaN/Inf. Но trailing-3 smoothed loss вышел за обязательные
±0.5% уже на пятой точке (`+0.5573%`), затем монотонно разошёлся до
`+7.3727%` на десятой. Поэтому быстрый результат не является победой и
[commit `4e61d99`](https://github.com/konard/uselessgoddess-quasar/commit/4e61d99b68a45db7dbdcb316359fc887444ded46)
откатил seam.

2K-step probe не продолжался после раннего нарушения gate: при измеренных
10.852–12.470 s/step полный paired run занял бы около 13 часов и не мог
изменить уже полученный отрицательный результат. Production остаётся fp32.

### P3: recomputed chunked cross-entropy — отклонён

[Run 30314563487](https://github.com/uselessgoddess/quasar/actions/runs/30314563487)
проверил следующий крупный рычаг на commit
[`82322a4`](https://github.com/konard/uselessgoddess-quasar/commit/82322a4f90a792bef6cb54a839d0413c20f3676d).
Custom backward не сохранял полный tensor logits: head пересчитывался блоками
по 256 позиций, softmax и все суммы оставались fp32. Отдельные unit tests
сравнили loss и оба gradient с materialized reference (`1e-5`), finite
difference (`2e-3`) и проверили отсутствие non-finite.

Три arm получили одинаковые seed, данные и 131 072 токена/step. После одного
warm-up измерялись девять steps, потому что autotune снова расширил окно:

| вариант | median | min/max | tok/s | effective | peak VRAM | решение |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| materialized, 4×32 | 12.365 s | 12.331–14.337 s | **10 600** | **5.13 TFLOP/s** | 14.360 GiB | reference |
| chunked-256, 4×32 | 13.763 s | 13.741–15.693 s | 9 524 | 4.61 TFLOP/s | **10.453 GiB** | revert |
| chunked-256, 8×16 | 30.966 s | 28.442–32.451 s | 4 233 | 2.05 TFLOP/s | **15.852 GiB** | reject |

Все десять напечатанных loss совпали между arms; максимальное отклонение
trailing-3 smoothed loss равно 0.0000%. Вариант 4×32 освободил 3.907 GiB, но
замедлил step на 10.15%. Попытка превратить этот запас в batch 8 замедлила
step на 60.07% и превысила обязательные 15 GiB на 0.852 GiB.

Причина — не численная: high-level graph режет `640×32768` head на множество
малых projection/recompute dispatch вместо одного хорошо загруженного GEMM.
Увеличение micro-batch дополнительно удваивает live activations остальных
слоёв. Поэтому [commit
`6a205ee`](https://github.com/konard/uselessgoddess-quasar/commit/6a205ee9b1c88ef57c6fabdf200e187665a30cbf)
откатил реализацию. Сохранение памяти само по себе не является throughput
победой; следующий head-кандидат должен сливать projection, softmax и backward
в одном или нескольких крупных native kernel.

### P2b: fp16 tied head с dynamic loss scaling — принят

[Run 30329788994](https://github.com/uselessgoddess/quasar/actions/runs/30329788994)
проверил f16-вариант минимального head seam на commit
[`2beee65`](https://github.com/konard/uselessgoddess-quasar/commit/2beee650ff17fff41764fccdfe60b0b36fdfa2dc).
В отличие от отклонённого bf16-варианта, loss перед backward умножался на 1024,
накопленные gradients один раз за optimizer step делились обратно в fp32, а
любой non-finite отменял update до обоих optimizers. Master weights, Muon/AdamW
state, hidden states, logits, softmax, z-loss и NLL оставались fp32.

Performance arms получили один warm-up и девять measured steps: известный по
P0 поздний autotune требовал расширенного окна, а spread первых трёх значений
снова оказался выше 3%. Harness теперь начинает с обязательной медианы трёх и
автоматически доходит до девяти только при таком spread. В таблице — медиана и
полный min/max исходного девятишагового evidence:

| вариант | samples | median | min/max | tok/s | effective | peak VRAM | решение |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fp32 control | 9 | 12.426 s | 12.400–14.414 s | 10 548 | 5.10 TFLOP/s | 14.111 GiB | reference |
| fp16 tied head | 9 | 10.802 s | 10.781–12.944 s | **12 134** | **5.87 TFLOP/s** | **12.111 GiB** | **production** |

FLOPs/token не менялись: выигрыш составил **15.04%**, а performance peak
снизился на 2.000 GiB. Отдельные quality arms использовали новый
детерминированный token stream на каждом accumulation pass, 20 measured steps
после warm-up и тот же seed. Trailing-3 loss сравнивался на каждой из 21 точек:
максимальное отклонение составило **0.1243%** при лимите 0.5%. На этом прогоне
fp16 arm дошёл до loss 0.0732; то есть короткий gate покрывает почти всю
сходимость синтетической learnable trajectory, а не только плоский старт.

20 steps выбраны вместо 200 осознанно. Одна paired quality точка стоит примерно
23 s; 200 steps потребовали бы около 77 минут только на две trajectory и вышли
бы за лимит одного короткого probe. Отклонённый bf16 seam нарушил gate уже на
пятой точке и к десятой разошёлся на 7.37%; новый горизонт вдвое длиннее этого
сигнала. Полный 200-step corpus/bpb A/B остаётся обязательным перед расширением
precision seam на несколько projection families.

Начальная диагностика совпала: activation min/max/mean
`-4.832604 / 5.017550 / 3.250181e-3`, grad norm `1.228238` у fp32 и `1.228284`
у fp16, loss `10.535172`, loss scale `1024`, non-finite count `0`. Из-за
дополнительных host-to-device copies varied-stream quality run использовал
больше памяти, чем performance arm, но его fp16 peak **13.892 GiB** также прошёл
15-GiB gate. `tiny-turbo` теперь выбирает этот head path по умолчанию; параметры
checkpoint остаются fp32, scaler state сохраняется рядом с номером шага, а
`--head-dtype fp32` оставляет явный escape hatch.

### P4: fp16 FFN projections — принят

[Run 30334108965](https://github.com/uselessgoddess/quasar/actions/runs/30334108965)
проверил следующее ровно одно projection family на commit
[`a10e66e`](https://github.com/konard/uselessgoddess-quasar/commit/a10e66ece54b1f90aa1524e7536e8da1373a44a1).
Reference уже использовал принятый fp16 head; candidate дополнительно переводил
в fp16 только `SwiGLU.inner`, `SwiGLU.outer` и `FFN.down` во всех 12 блоках.
Master weights, RMSNorm, SiLU, elementwise multiply, residual stream, logits и
loss оставались fp32.

Оба performance arm автоматически расширились с трёх до девяти samples из-за
первого позднего autotune:

| вариант | samples | median | min/max | tok/s | effective | peak VRAM | решение |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fp16 head | 9 | 10.792 s | 10.774–12.864 s | 12 146 | 5.87 TFLOP/s | 12.112 GiB | reference |
| fp16 head + FFN | 9 | 8.929 s | 8.922–11.036 s | **14 679** | **7.10 TFLOP/s** | **12.038 GiB** | **production** |

FLOPs/token не менялись: дополнительный выигрыш составил **20.85%** поверх
принятого head path, а performance peak уменьшился ещё на 0.074 GiB. На
отдельных 21-point varied-token trajectories максимальное trailing-3
loss-отклонение составило только **0.0060%** при лимите 0.5%; обе стороны
завершили на loss 0.0732. Все 42 training points сохранили loss scale 1024.
Диагностика reference/candidate: activation
`[-4.832604, 5.017550], mean 3.250181e-3` против
`[-4.832693, 5.017528], mean 3.249888e-3`, одинаковый grad norm `1.228284`,
loss `10.535172` и non-finite count `0`. Quality peak candidate 12.059 GiB
тоже прошёл 15-GiB gate.

Численный unit test отдельно сравнивает fp32/fp16 forward, input gradient и
inner-weight gradient с допуском `1e-2`, проверяет fp32 master weight/output и
масштабированный backward. `tiny-turbo` включает оба измеренных seam по
умолчанию; `--head-dtype fp32` и `--ffn-dtype fp32` отключают их независимо.

### P5: fp16 Mamba input/output projections — принят

[Run 30338965261](https://github.com/uselessgoddess/quasar/actions/runs/30338965261)
проверил последнюю разрешённую model-level projection family на commit
[`bb67875`](https://github.com/konard/uselessgoddess-quasar/commit/bb67875e2326d4076e417a86ab0f1491e414c264).
Reference использовал принятые fp16 head+FFN, а candidate дополнительно
переводил в fp16 только Mamba `in_proj` и `out_proj`. SSD coefficients,
discretization, recurrent state, norms, gating, residual stream, master
weights, optimizer state, logits и loss оставались fp32.

Оба performance arm автоматически расширились с трёх до девяти samples из-за
spread выше 3%:

| вариант | samples | median | min/max | tok/s | effective | peak VRAM | решение |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fp16 head + FFN | 9 | 8.982 s | 8.973–11.061 s | 14 593 | 7.06 TFLOP/s | 12.038 GiB | reference |
| fp16 head + FFN + Mamba | 9 | 7.408 s | 7.406–9.511 s | **17 693** | **8.56 TFLOP/s** | **11.835 GiB** | **production** |

Дополнительный выигрыш составил **21.24%**, а performance peak уменьшился ещё
на 0.203 GiB. На отдельных 21-point varied-token trajectories максимальное
trailing-3 loss-отклонение снова составило **0.0060%** при лимите 0.5%; обе
стороны завершили на loss 0.0732. Все 62 training points четырёх arms
сохранили loss scale 1024. Диагностика reference/candidate: activation
`[-4.832693, 5.017528], mean 3.249888e-3` против
`[-4.831584, 5.017357], mean 3.248936e-3`, grad norm `1.228284` против
`1.228251`, loss `10.535172` против `10.535189`, non-finite count `0`.
Quality peak candidate 11.835 GiB также прошёл 15-GiB gate.

Численный unit test сравнивает full forward, recurrent step, input gradient и
обе projection-weight gradients с fp32 reference, проверяет fp32 output/master
weights и масштабированный backward. `tiny-turbo` включает все три измеренных
seam по умолчанию; `--head-dtype fp32`, `--ffn-dtype fp32` и
`--mamba-dtype fp32` отключают их независимо.

### P6: ROCm rocWMMA backend — отклонён

[Run 30345853678](https://github.com/uselessgoddess/quasar/actions/runs/30345853678)
изолировал backend на крупнейшей projection-форме модели: bias-free
`4096×640 @ 640×32768`, forward и оба GEMM backward, fp16 operands с fp32
masters/outputs и loss scale 1024. Vulkan и ROCm rocWMMA получили одинаковые
данные, один warm-up и median трёх samples; spread был ниже 3%, поэтому
auto-extension до девяти не понадобился.

| backend | median | min/max | throughput | peak VRAM | решение |
| --- | ---: | ---: | ---: | ---: | --- |
| Vulkan | 20.006 ms | 19.930–20.425 ms | **25.76 TFLOP/s** | **2.277 GiB** | reference |
| ROCm rocWMMA | 260.579 ms | 260.014–262.950 ms | **1.98 TFLOP/s** | **6.799 GiB** | reject |

rocWMMA оказался на **92.31% медленнее** reference, хотя прошёл memory и
численные gates: activation `[-0.181641, 0.290039]`, mean `1.917477e-6`,
grad norm `2.449194e-2`, non-finite count `0`, максимальная forward ошибка
`1.220703e-4`, ошибки gradients не выше `1.644075e-7`.

Первый запуск не дошёл до matmul из-за `hipMemcpy2DAsync` с
`hipErrorInvalidValue` на contiguous rank-2 upload. Повтор на commit
[`2f089ca`](https://github.com/konard/uselessgoddess-quasar/commit/2f089ca02169748609cfab5b931b111aa858ccf8)
передал те же contiguous bytes через rank-1 и zero-copy reshape, после чего
измерил настоящий rocWMMA kernel. CubeCL позже исправил тот же staging defect
в upstream commit
[`174238e`](https://github.com/tracel-ai/cubecl/commit/174238e23f5d1334082db37291684f230130c37b),
но transfer fix не меняет измеренный 13× разрыв kernel throughput. Поэтому
rocWMMA feature удалён; следующий независимый backend gate использует
hipBLASLt напрямую.

### P7: direct hipBLASLt linear — прошёл изолированный gate

[Run 30352703527](https://github.com/uselessgoddess/quasar/actions/runs/30352703527)
измерил hipBLASLt без CubeCL ROCm compiler на той же крупнейшей
projection-форме `4096×640 @ 640×32768`, включая forward и оба GEMM backward.
Reference и candidate использовали fp16 operands, fp32 outputs/gradient
accumulation, loss scale 1024 и один warm-up. Spread первых трёх samples
превысил 3% в обеих сторонах, поэтому harness автоматически расширил каждое
окно до девяти.

| backend | median | min/max | throughput | peak VRAM | решение |
| --- | ---: | ---: | ---: | ---: | --- |
| Vulkan | 19.124 ms | 18.808–19.497 ms | 26.95 TFLOP/s | 2.285 GiB | reference |
| direct hipBLASLt | 5.309 ms | 4.943–5.614 ms | **97.07 TFLOP/s** | **1.765 GiB** | production integration probe |

Vendor library оказался на **260.19% быстрее** reference и прошёл 15-GiB
gate. Для hipBLASLt activation был `[-0.198242, 0.198364]`, mean
`1.716217e-9`, grad norm `9.727625e-2`, non-finite count `0`; максимальные
ошибки output, input gradient и weight gradient относительно reference равны
`0`. Исправленный source commit
[`d7d4bd4`](https://github.com/konard/uselessgoddess-quasar/commit/d7d4bd4c538bd11c2c963b9d81980d714e5ea16a)
оставляет этот exact-shape probe постоянным CI gate.

Это локальный roofline, а не full-step claim: он доказал, что vendor GEMM
достаточно быстр для следующей пробы, но не учитывал остальную модель,
синхронизацию между runtimes и optimizer trajectory.

### P8: production hipBLASLt tied head — отклонён

[Run 30360553662](https://github.com/uselessgoddess/quasar/actions/runs/30360553662)
проверил production tied-head integration на commit
[`d0695f3`](https://github.com/konard/uselessgoddess-quasar/commit/d0695f37cdb6a122660c6592aa26f904336d3821).
Native forward и оба gradients сначала совпали с обычным ROCm matmul в
отдельном regression test. Но RDNA4 path требовал незавершённые
[Burn PR #5188](https://github.com/tracel-ai/burn/pull/5188) и
[CubeCL PR #1437](https://github.com/tracel-ai/cubecl/pull/1437): без них
throughput calibration переполнял 32-bit launch scalar до первого native
вызова.

Обязательный full optimizer-step A/B остановился ещё на Vulkan control с этим
dependency pair, до запуска hipBLASLt candidate. Начальная диагностика была
finite: activation `[-6.394674, 20.93708]`, mean `3.229315e-3`, grad norm
`1.226088`, loss `10.535510`, loss scale 1024 и non-finite count `0`. После
warm-up control измерил один step за 8.438 s (**15 534 tok/s**), а следующий
step получил non-finite gradients и был немедленно остановлен dynamic scaler.
Peak VRAM **12.153 GiB** прошёл memory gate, но numerical gate — нет.

Поскольку reference сам нарушил обязательное условие finite trajectory,
валидного paired throughput или quality результата для production hipBLASLt
не существует. Native integration, незавершённые dependency revisions и
связанный рост MSRV удалены одним rejection commit; принятый P5 Vulkan path
восстановлен без изменений. Изолированный P7 harness остаётся как
воспроизводимое свидетельство и gate для будущей стабильной интеграции.

### Итог P0–P8 и ETA

Полный default `tiny-turbo` содержит 1.6384B токенов. Строка P1 — отдельный
GEMM roofline, поэтому она не подменяет full-step throughput:

| стадия | tok/s | effective | peak VRAM | ETA default | статус |
| --- | ---: | ---: | ---: | ---: | --- |
| P0 frozen fp32 | 10 558 | 5.11 TFLOP/s | 14.516 GiB | 43.1 ч | production |
| P1 bf16 8192³ GEMM | — | 43.75 TFLOP/s roofline | — | — | stable diagnostic |
| P2 bf16 tied head | 12 078 | 5.84 TFLOP/s | 13.359 GiB | 37.7 ч | quality fail |
| P3 chunked CE, 4×32 | 9 524 | 4.61 TFLOP/s | 10.453 GiB | 47.8 ч | throughput fail |
| P3 chunked CE, 8×16 | 4 233 | 2.05 TFLOP/s | 15.852 GiB | 107.5 ч | VRAM + throughput fail |
| P2b f16 tied head | 12 134 | 5.87 TFLOP/s | 12.111 GiB | 37.5 ч | accepted |
| P4 f16 head + FFN | 14 679 | 7.10 TFLOP/s | 12.038 GiB | 31.0 ч | accepted |
| P5 f16 head + FFN + Mamba | **17 693** | **8.56 TFLOP/s** | **11.835 GiB** | **25.7 ч** | production |
| P6 ROCm rocWMMA linear | — | 1.98 TFLOP/s isolated | 6.799 GiB | — | rejected |
| P7 direct hipBLASLt linear | — | 97.07 TFLOP/s isolated | 1.765 GiB | — | integration probe passed |
| P8 production hipBLASLt tied head | — | no valid paired result | 12.153 GiB control | — | non-finite control; rejected |

`examples/performance_baseline.sh` теперь не только сохраняет sampler log, но
и валит CI, если production peak превышает 15 GiB. Он запускает принятые fp16
head + FFN + Mamba paths; `examples/fp16_head_ab.sh`,
`examples/fp16_ffn_ab.sh` и `examples/fp16_mamba_projection_ab.sh` сохраняют
каждый независимый paired control.

### Stop/pivot analysis

Цель 40 TFLOP/s всё ещё требует 4.7× к принятому P5. Текущий stack пока не
показывает такого запаса:

- full step после трёх принятых precision seams использует 8.56 TFLOP/s,
  тогда как изолированный fp32 GEMM даёт 14.05 TFLOP/s, а f16 GEMM —
  42–43 TFLOP/s: значительная часть шага всё ещё не превращается в крупные
  matrix kernels;
- matmul занимает 67.4% GPU timestamps, но recorded kernels покрывают только
  392 ms из 1.040 s wall time профилированного steady micro-batch. Около 62%
  wall time остаётся в launch/queue/orchestration;
- Vulkan bf16 GEMM достигает 43–44 TFLOP/s, лишь 44–45% номинального matrix
  peak. Даже идеальное превращение всей модели в этот roofline почти не
  оставляет бюджета на softmax, norms, SSD и optimizer;
- локальные head, FFN и Mamba probes доказали, что WMMA ускоряет реальную
  forward/backward нагрузку. Bf16 head без scaling нарушил trajectory, но f16
  с dynamic scaling прошёл head и обе projection families: dtype и safeguard
  нельзя объединять в один вывод;
- recomputed CE доказал точное совпадение trajectory и освободил 3.907 GiB, но
  high-level chunk graph оказался на 10.15% медленнее, а попытка использовать
  запас для batch 8 — на 60.07% медленнее и выше VRAM gate.
- rocWMMA корректно выполнил крупнейшую projection-форму, но дал только
  1.98 TFLOP/s против 25.76 TFLOP/s Vulkan и был отклонён до full-step пробы.
- direct hipBLASLt выполнил ту же projection-форму с 97.07 TFLOP/s против
  26.95 TFLOP/s Vulkan, но production integration потребовал незавершённого
  Burn/CubeCL stack; на нём сам Vulkan control получил non-finite gradients до
  запуска candidate.

Принятые P2b/P4/P5 одновременно быстры, finite и находятся внутри loss gate,
но P5 всё же остался ниже 20 effective TFLOP/s. Поэтому model-level precision
и мелкие elementwise fusion прекращаются; глобального autocast по-прежнему
нет. Приоритет переходит к fused head/CE kernel с fp32 softmax и gradient
accumulation. Следующая production hipBLASLt/ROCm попытка должна сначала
получить stable Burn/CubeCL revision, который повторяет finite P5 control без
изменения accepted Vulkan dependency baseline.
Изменение vocab относится к отдельному quality/GPU-hour эксперименту и не
может объявляться победой по raw tok/s.

Этот отдельный эксперимент был принят в issue #23 — но именно как
memory/quality-решение, а не как результат из этой таблицы: `tiny` и
`tiny-turbo` перешли на 8192 (`DESIGN.md` §2.6). Все числа выше измерены на
32768 и здесь не пересчитываются: форма head GEMM в них — `4096×640 @
640×32768`, а на новом словаре это `4096×640 @ 640×8192`. Ранжирование
кандидатов от этого не меняется (все arms сдвигаются одинаково), но абсолютные
tok/s и effective TFLOP/s на новом словаре надо перемерить, прежде чем
сравнивать с ними.

## Precision и profiler

На Vulkan глобальный bf16 был отвергнут backend'ом, а глобальный f16 попал в
panic Burn fusion и получил non-finite loss. На ROCm оба reduced dtype ранее
падали при выборе RDNA4 WMMA. Эти global-dtype пробы не противоречат
P2b/P4/P5: production device, master weights и остальная модель остаются fp32,
а только измеренные head, FFN и Mamba projection GEMM явно cast'ятся в f16 и
защищены scaler. Решение основано на paired измерениях, а не на предположении
об AMP.

CubeCL profiler насчитал 9 948 kernel launches даже для одного micro-batch.
Matmul занимает около 79% записанного GPU time, а elementwise/reduce/slice/copy
дают тысячи коротких dispatch. Fused rank-one Mamba-3 forward с custom backward
убрал часть этого overhead и дал ещё 5.6% поверх K4, но оставшийся разрыв нельзя
закрыть локальной заменой RMSNorm. Профилирование сериализует launches, и его
266 tok/s не является throughput-замером.

## Wall-clock

Для формы из issue (`4 × 12 × 1024 = 49152` токена/step) 12 500 шагов — 614.4M
токенов. Скриншотные 5 966 tok/s означают 28.6 часа; P4 при 14 679 tok/s —
11.6 часа, а P5 при 17 693 tok/s — **9.6 часа**. Цель 9 часов требует
18 963 tok/s, 8 часов — 21 333 tok/s. Текущий результат сокращает исходный
запуск примерно на 19 часов, но до заявленного overnight target всё ещё нужен
выигрыш 1.07–1.21×.

Production default обрабатывает 1.6384B токенов за те же 12 500 шагов, поэтому
при 17 693 tok/s занимает около **25.7 часа**. Для короткого issue-budget можно
явно задать `--accum 12`; startup сразу покажет, что полный token budget
изменился.

Сравнивать модели и реализации нужно по tok/s при одинаковом числе токенов,
effective TFLOP/s и bits-per-byte. Число optimizer steps без batch shape этих
трёх вопросов не отвечает.
