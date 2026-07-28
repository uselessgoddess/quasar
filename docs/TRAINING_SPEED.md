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
3. Burn/burn-mamba в закреплённых ревизиях не имеют AMP. Обучение идёт в fp32;
   pure bf16/f16 не является безопасной заменой autocast + GradScaler.
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

Итоговая production-форма с fused scan дала **10 578 tok/s** при пике 14.074
GiB. `tiny-turbo` поэтому выбирает 640 × 12, micro-batch 4, `serial`,
checkpointing off и один CubeCL stream. Accumulation по умолчанию поднят до
32: effective batch остаётся `4 × 32 × 1024 = 131072`, то есть скорость не
куплена сокращением обучения.

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

### Итог P0–P3 и ETA

Полный default `tiny-turbo` содержит 1.6384B токенов. Строка P1 — отдельный
GEMM roofline, поэтому она не подменяет full-step throughput:

| стадия | tok/s | effective | peak VRAM | ETA default | статус |
| --- | ---: | ---: | ---: | ---: | --- |
| P0 frozen fp32 | 10 558 | 5.11 TFLOP/s | 14.516 GiB | 43.1 ч | production |
| P1 bf16 8192³ GEMM | — | 43.75 TFLOP/s roofline | — | — | stable diagnostic |
| P2 bf16 tied head | 12 078 | 5.84 TFLOP/s | 13.359 GiB | 37.7 ч | quality fail |
| P3 chunked CE, 4×32 | 9 524 | 4.61 TFLOP/s | 10.453 GiB | 47.8 ч | throughput fail |
| P3 chunked CE, 8×16 | 4 233 | 2.05 TFLOP/s | 15.852 GiB | 107.5 ч | VRAM + throughput fail |
| production после P3 | **10 558** | **5.11 TFLOP/s** | **14.516 GiB** | **43.1 ч** | fp32 |

`examples/performance_baseline.sh` теперь не только сохраняет sampler log, но
и валит CI, если production peak превышает 15 GiB.

### Stop/pivot analysis

Цель 40 TFLOP/s требует почти 7.8× к P0. Текущий stack не показывает такого
запаса:

- full step использует 5.11 TFLOP/s, тогда как изолированный fp32 GEMM даёт
  14.05 TFLOP/s: orchestration и не-GEMM части оставляют ещё примерно 2.7×
  между моделью и уже достигнутым fp32 roofline;
- matmul занимает 67.4% GPU timestamps, но recorded kernels покрывают только
  392 ms из 1.040 s wall time профилированного steady micro-batch. Около 62%
  wall time остаётся в launch/queue/orchestration;
- Vulkan bf16 GEMM достигает 43–44 TFLOP/s, лишь 44–45% номинального matrix
  peak. Даже идеальное превращение всей модели в этот roofline почти не
  оставляет бюджета на softmax, norms, SSD и optimizer;
- локальный head probe доказал, что WMMA ускоряет реальную forward/backward
  нагрузку, но обычный autodiff через bf16 cast не сохраняет требуемую
  optimization trajectory. Расширять тот же seam на FFN/QKV/Mamba projections
  после такого сигнала небезопасно;
- recomputed CE доказал точное совпадение trajectory и освободил 3.907 GiB, но
  high-level chunk graph оказался на 10.15% медленнее, а попытка использовать
  запас для batch 8 — на 60.07% медленнее и выше VRAM gate.

P2 и P3 не достигли даже 20 effective TFLOP/s и оба откатились. P4 (более
широкий mixed precision) не запускается после провала минимального P2 seam:
он увеличивает уже измеренное расхождение trajectory. P5–P7 также не могут
честно закрыть почти 7.8× локальными fusion или изменением model contract.
Здесь срабатывает обязательный stop/pivot.

Следующий допустимый go/no-go — не ещё один high-level cast или chunk graph.
Он должен дать для одного `640×32768` head fused bf16 forward/backward с fp32
master weight, fp32 softmax/gradient accumulation, стабильность без non-finite
и крупные dispatch. Приоритетный путь:

1. custom CubeCL kernel, объединяющий projection, fp32 cross-entropy и VJP без
   materialized logits и без 256-row matmul graph;
2. isolated kernel gate против measured bf16 roofline, затем те же numerical,
   VRAM и paired full-step проверки;
3. продолжать backend work только если full step превысит
   20 effective TFLOP/s;
4. если Vulkan не проходит fused-head go/no-go, timeboxed HIP/hipBLASLt или Burn
   ROCm backend spike. Не продолжать мелкие elementwise fusion до этого
   решения.

## Precision и profiler

На Vulkan bf16 отвергнут backend'ом, а f16 попал в panic Burn fusion и получил
non-finite loss. На ROCm оба reduced dtype ранее падали при выборе RDNA4 WMMA.
Эти старые global-dtype пробы не противоречат новому roofline: точечный cast
вокруг GEMM теперь работает. Production всё равно остаётся fp32, потому что
первый такой full-step seam не прошёл loss gate. Решение основано на paired
измерении, а не на предположении об AMP.

CubeCL profiler насчитал 9 948 kernel launches даже для одного micro-batch.
Matmul занимает около 79% записанного GPU time, а elementwise/reduce/slice/copy
дают тысячи коротких dispatch. Fused rank-one Mamba-3 forward с custom backward
убрал часть этого overhead и дал ещё 5.6% поверх K4, но оставшийся разрыв нельзя
закрыть локальной заменой RMSNorm. Профилирование сериализует launches, и его
266 tok/s не является throughput-замером.

## Wall-clock

Для формы из issue (`4 × 12 × 1024 = 49152` токена/step) 12 500 шагов — 614.4M
токенов. Скриншотные 5 966 tok/s означают 28.6 часа; финальные 10 368 tok/s —
**16.5 часа**. Цель 9 часов требует 18 963 tok/s, 8 часов — 21 333 tok/s.
Текущий результат сокращает запуск примерно на 12.1 часа, но до заявленного
overnight target всё ещё нужен выигрыш 1.8–2.1×.

Production default обрабатывает 1.6384B токенов за те же 12 500 шагов, поэтому
при 10 578 tok/s занимает около **43.0 часа**. Для короткого issue-budget можно
явно задать `--accum 12`; startup сразу покажет, что полный token budget
изменился.

Сравнивать модели и реализации нужно по tok/s при одинаковом числе токенов,
effective TFLOP/s и bits-per-byte. Число optimizer steps без batch shape этих
трёх вопросов не отвечает.
