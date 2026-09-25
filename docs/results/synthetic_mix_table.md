# synthetic_mix: simulated deadline-miss rate (%) vs utilization

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
| 0.30 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | admit | admit | admit |
| 0.40 | 0.09 | 0.03 | 0.01 | 0.01 | 0.00 | 0.01 | 0.04 | admit | admit | admit |
| 0.50 | 0.39 | 0.26 | 0.09 | 0.09 | 0.03 | 0.08 | 0.27 | admit | admit | reject |
| 0.60 | 2.11 | 1.87 | 0.38 | 0.32 | 0.21 | 0.22 | 1.32 | reject | admit | reject |
| 0.70 | 4.07 | 3.40 | 0.63 | 0.62 | 0.63 | 0.50 | 3.03 | reject | reject | reject |
| 0.80 | 8.53 | 8.38 | 1.25 | 1.10 | 0.75 | 0.67 | 4.53 | reject | reject | reject |
| 0.90 | 17.47 | 18.59 | 2.52 | 2.12 | 1.27 | 1.58 | 6.35 | reject | reject | reject |
| 1.00 | 24.62 | 26.10 | 5.34 | 5.50 | 1.79 | 2.69 | 7.80 | reject | reject | reject |
| 1.10 | 32.89 | 33.04 | 11.76 | 20.13 | 3.37 | 5.55 | 11.54 | reject | reject | reject |
| 1.20 | 43.11 | 43.40 | 22.29 | 35.35 | 5.15 | 16.51 | 18.03 | reject | reject | reject |
| 1.30 | 53.63 | 53.01 | 32.89 | 47.51 | 8.85 | 30.37 | 24.57 | reject | reject | reject |

## High-criticality streams

| U | round_robin | fifo | fixed_priority | edf | edf_batch | edf_batch_same_deadline | dynamic_batch | test: none | test: same_deadline | test: cross_deadline |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|
| 0.30 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | admit | admit | admit |
| 0.40 | 0.04 | 0.01 | 0.02 | 0.02 | 0.00 | 0.02 | 0.06 | admit | admit | admit |
| 0.50 | 0.37 | 0.20 | 0.13 | 0.13 | 0.04 | 0.11 | 0.34 | admit | admit | reject |
| 0.60 | 2.60 | 2.39 | 0.54 | 0.48 | 0.30 | 0.32 | 1.78 | reject | admit | reject |
| 0.70 | 4.85 | 4.23 | 0.90 | 0.91 | 0.93 | 0.74 | 4.15 | reject | reject | reject |
| 0.80 | 10.47 | 10.87 | 1.63 | 1.58 | 1.02 | 0.91 | 6.22 | reject | reject | reject |
| 0.90 | 22.64 | 24.99 | 2.70 | 2.79 | 1.70 | 2.19 | 8.57 | reject | reject | reject |
| 1.00 | 31.82 | 34.49 | 5.00 | 6.60 | 2.46 | 3.43 | 10.36 | reject | reject | reject |
| 1.10 | 42.21 | 42.75 | 10.34 | 18.64 | 4.56 | 6.28 | 15.14 | reject | reject | reject |
| 1.20 | 54.83 | 55.35 | 18.63 | 32.10 | 6.57 | 16.84 | 23.73 | reject | reject | reject |
| 1.30 | 66.61 | 66.84 | 27.36 | 45.60 | 11.11 | 27.13 | 31.92 | reject | reject | reject |
