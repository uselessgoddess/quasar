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

Low-precision ошибки сохраняются как результат probe, но fp32 обязан пройти.
Это не mixed-precision training: tensors явно cast'ятся вокруг одного GEMM,
пока default dtype устройства остаётся fp32. Такой seam соответствует
планируемому точечному пути для крупных projection layers.

## Результаты

Таблица заполняется только измерениями self-hosted RX 9070 XT с точным commit,
driver/runtime и min/max из CI artifact.

| backend | dtype | shape | median TFLOP/s | min/max | peak utilization | stability |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Vulkan | fp32 | 4096³ | pending | pending | pending | pending |
| Vulkan | fp32 | 8192³ | pending | pending | pending | pending |
| Vulkan | f16 | 4096³ | pending | pending | pending | pending |
| Vulkan | f16 | 8192³ | pending | pending | pending | pending |
| Vulkan | bf16 | 4096³ | pending | pending | pending | pending |
| Vulkan | bf16 | 8192³ | pending | pending | pending | pending |

До заполнения этой таблицы precision-кандидат не получает performance claim.
