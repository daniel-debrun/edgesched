# vla_plus_control: simulated deadline-miss rate (%) vs utilization

Simulated, not measured. Mean over seeds. The `test:` columns are the
non-preemptive EDF admission verdicts for the scaled task set under each
batching rule: `none` certifies `edf`, `same_deadline` certifies
`edf_batch_same_deadline`, `cross_deadline` certifies `edf_batch`. The tests
use nominal latencies while the simulator adds execution-time noise, so an
admitted row can still show small miss rates. They say nothing about FIFO,
round-robin, fixed priority or the Triton-style batcher.

## All streams

| U | round_robin | fifo | fixed_priority | edf | edf_batch | edf_batch_same_deadline | dynamic_batch | test: none | test: same_deadline | test: cross_deadline |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|
| 0.30 | 9.13 | 9.15 | 9.13 | 9.15 | 9.15 | 9.15 | 9.15 | reject | reject | reject |
| 0.40 | 12.47 | 12.48 | 12.54 | 12.48 | 12.48 | 12.48 | 12.48 | reject | reject | reject |
| 0.50 | 16.02 | 16.02 | 16.12 | 16.02 | 16.02 | 16.02 | 16.02 | reject | reject | reject |
| 0.60 | 19.28 | 19.19 | 19.51 | 19.19 | 19.19 | 19.19 | 19.19 | reject | reject | reject |
| 0.70 | 22.69 | 22.58 | 22.70 | 22.58 | 22.58 | 22.58 | 22.58 | reject | reject | reject |
| 0.80 | 26.17 | 25.78 | 25.87 | 25.78 | 25.78 | 25.78 | 25.78 | reject | reject | reject |
| 0.90 | 29.55 | 29.16 | 29.97 | 29.16 | 29.16 | 29.16 | 29.16 | reject | reject | reject |
| 1.00 | 33.06 | 33.10 | 33.38 | 33.10 | 33.10 | 33.10 | 33.10 | reject | reject | reject |
| 1.10 | 36.50 | 36.26 | 36.68 | 36.26 | 36.26 | 36.26 | 36.26 | reject | reject | reject |
| 1.20 | 40.06 | 40.18 | 40.01 | 40.18 | 40.18 | 40.18 | 40.18 | reject | reject | reject |
| 1.30 | 43.66 | 44.83 | 43.19 | 44.83 | 44.83 | 44.83 | 44.83 | reject | reject | reject |

## High-criticality streams

| U | round_robin | fifo | fixed_priority | edf | edf_batch | edf_batch_same_deadline | dynamic_batch | test: none | test: same_deadline | test: cross_deadline |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|
| 0.30 | 9.19 | 9.21 | 9.19 | 9.21 | 9.21 | 9.21 | 9.21 | reject | reject | reject |
| 0.40 | 12.55 | 12.56 | 12.62 | 12.56 | 12.56 | 12.56 | 12.56 | reject | reject | reject |
| 0.50 | 16.13 | 16.12 | 16.23 | 16.12 | 16.12 | 16.12 | 16.12 | reject | reject | reject |
| 0.60 | 19.41 | 19.32 | 19.64 | 19.32 | 19.32 | 19.32 | 19.32 | reject | reject | reject |
| 0.70 | 22.84 | 22.73 | 22.85 | 22.73 | 22.73 | 22.73 | 22.73 | reject | reject | reject |
| 0.80 | 26.34 | 25.96 | 26.04 | 25.96 | 25.96 | 25.96 | 25.96 | reject | reject | reject |
| 0.90 | 29.75 | 29.36 | 30.17 | 29.36 | 29.36 | 29.36 | 29.36 | reject | reject | reject |
| 1.00 | 33.28 | 33.32 | 33.60 | 33.32 | 33.32 | 33.32 | 33.32 | reject | reject | reject |
| 1.10 | 36.74 | 36.50 | 36.93 | 36.50 | 36.50 | 36.50 | 36.50 | reject | reject | reject |
| 1.20 | 40.33 | 40.44 | 40.28 | 40.44 | 40.44 | 40.44 | 40.44 | reject | reject | reject |
| 1.30 | 43.95 | 45.13 | 43.48 | 45.13 | 45.13 | 45.13 | 45.13 | reject | reject | reject |
