# GEMM roofline на RX 9070 XT

Этот документ фиксирует обязательный gate перед precision-оптимизациями
`tiny-turbo`. Замер идёт через тот же Burn/CubeCL/Vulkan stack, что и обучение,
а не через отдельную библиотеку: именно этот путь должен реально выбрать WMMA
и пережить синхронизацию на `gfx1201`.

## Методика

[`examples/roofline.rs`](../examples/roofline.rs) создаёт входы вне измеряемого
участка, делает один warm-up, затем минимум три отдельных synchronized GEMM.
Результат — медиана; всегда печатаются min/max. Если разброс min/max превышает
3%, окно автоматически расширяется до девяти измерений.

Номинальные пики, используемые только для колонки utilization:

- fp32: 48 TFLOP/s;
- fp16/bf16 matrix: 97 TFLOP/s.

В CI запускаются квадратные GEMM 4096 и 8192 для fp32, f16 и bf16:

```sh
examples/roofline.sh
```

Ошибка любой отдельной пары dtype/shape сохраняется как результат probe. Хотя
бы одна fp32-форма обязана пройти, иначе результаты reduced precision нельзя
интерпретировать. Это не mixed-precision training: tensors явно cast'ятся
вокруг одного GEMM, пока default dtype устройства остаётся fp32. Такой seam
соответствует планируемому точечному пути для крупных projection layers.

## Результаты

Измерение: [run 30310985422](https://github.com/uselessgoddess/quasar/actions/runs/30310985422),
source commit `fc0e5f8746905bb931fcc1be21eb9dd849d6d05f` (PR merge
`b9ab44709f9a39b93f4b335ac7b0b5fa1beaff9c`), RX 9070 XT `gfx1201`, RADV
Mesa 26.1.5, Vulkan 1.4.354, Rust 1.97.1. Burn закреплён на `d028234e`, CubeCL
на `3beb9afa`.

| backend | dtype | shape | median TFLOP/s | min/max | peak utilization | stability |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Vulkan/RADV | fp32 | 4096³ | 14.05 | 13.90–14.18 | 29.3% | 3 samples, spread 2.03% |
| Vulkan/RADV | fp32 | 8192³ | — | — | — | device lost на warm-up |
| Vulkan/RADV | f16 | 4096³ | 42.03 | 39.11–43.00 | 43.3% | 9 samples, spread 9.95% |
| Vulkan/RADV | f16 | 8192³ | 43.25 | 42.64–43.39 | 44.6% | 3 samples, spread 1.77% |
| Vulkan/RADV | bf16 | 4096³ | 41.49 | 39.45–43.89 | 42.8% | 9 samples, spread 11.27% |
| Vulkan/RADV | bf16 | 8192³ | 43.75 | 43.41–43.99 | 45.1% | 3 samples, spread 1.33% |

fp32 8192³ воспроизводимо дошёл до hard recovery RADV, после чего wgpu сообщил
`Parent device is lost`; отдельные процессы следующих проб не пострадали. Это
не основание скрывать форму или валить всю матрицу: fp32 4096³ остаётся
валидным reference, а failed shape явно записан в таблицу.

## Решение gate

Обе точечные 16-битные формы работают, примерно в 3 раза быстрее fp32 4096³.
bf16 выбран первым кандидатом: на большой форме он чуть быстрее f16 и сохраняет
динамический диапазон fp32, поэтому не требует loss scaling. Следующий тест —
не глобальный dtype, а explicit casts только вокруг tied output-head GEMM с
fp32 master weight, logits/loss и optimizer state.

Этот downstream gate выполнен в
[run 30312309297](https://github.com/uselessgoddess/quasar/actions/runs/30312309297).
Head действительно ускорил полный step на 14.9% и снизил peak VRAM, но
trailing-3 loss разошёлся на 7.37% при лимите 0.5%. Seam откатан: работающий
bf16 roofline доказывает доступность matrix path, но не доказывает безопасный
обычный autodiff через reduced-precision casts. Следующий go/no-go требует
custom backward с fp32 gradient accumulation.

Следующий downstream gate тоже выполнен в
[run 30314563487](https://github.com/uselessgoddess/quasar/actions/runs/30314563487).
Точный recomputed cross-entropy блоками по 256 позиций совпал с materialized
fp32 reference по loss и gradients, уменьшил peak VRAM с 14.360 до 10.453
GiB, но замедлил полный step с 5.13 до 4.61 TFLOP/s (−10.15%). Batch 8×16
дал только 2.05 TFLOP/s и 15.852 GiB, поэтому одновременно нарушил throughput
и memory gates.

Этот результат уточняет roofline-разрыв: убрать materialized logits
недостаточно. High-level chunks превращают большой head GEMM в множество
малых projection/recompute dispatch, тогда как roofline измеряет крупную,
непрерывно загруженную матрицу. P3 откатан. Следующий go/no-go должен быть
настоящим fused head kernel (projection + fp32 softmax/loss + VJP), а не
chunking существующего autodiff graph; после двух кандидатов ниже 20
effective TFLOP/s дальнейшие локальные fusion остановлены до такого backend
spike.

После просьбы продолжить точечный
[f16 gate](https://github.com/uselessgoddess/quasar/actions/runs/30329788994)
изменил этот вывод. С dynamic loss scale 1024 тот же tied head сохранил fp32
master/logits/loss, дал 12 134 против 10 548 tok/s (+15.04%), снизил
performance peak VRAM с 14.111 до 12.111 GiB и удержал максимальное trailing-3
loss-отклонение на 0.1243% при лимите 0.5%. F16 head принят в `tiny-turbo`;
глобальный dtype по-прежнему не используется.

Следующий downstream gate расширяет f16 ровно на одно крупное projection
family. Если этот контролируемый P4 не приближает full step к обязательным
20 effective TFLOP/s, решение возвращается к fused head/CE или backend spike,
а не к дальнейшему бесконтрольному autocast.
