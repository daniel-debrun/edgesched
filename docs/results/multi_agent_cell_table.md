# multi_agent_cell: simulated deadline-miss rate (%) vs utilization

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
| 0.30 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | admit | admit | admit |
| 0.40 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | admit | admit | admit |
| 0.50 | 0.09 | 0.09 | 0.00 | 0.00 | 0.00 | 0.00 | 0.17 | admit | admit | admit |
| 0.60 | 3.92 | 3.92 | 0.01 | 0.01 | 0.00 | 0.00 | 4.81 | admit | admit | admit |
| 0.70 | 8.16 | 8.16 | 0.05 | 0.05 | 0.03 | 0.04 | 8.54 | admit | admit | reject |
| 0.80 | 8.66 | 8.64 | 0.60 | 0.60 | 0.04 | 0.04 | 8.70 | reject | admit | reject |
| 0.90 | 9.04 | 9.11 | 6.20 | 6.18 | 0.20 | 0.24 | 9.43 | reject | reject | reject |
| 1.00 | 17.82 | 15.57 | 13.10 | 13.86 | 0.64 | 0.63 | 10.11 | reject | reject | reject |
| 1.10 | 26.66 | 26.99 | 23.70 | 32.86 | 6.77 | 7.66 | 16.29 | reject | reject | reject |
| 1.20 | 43.87 | 41.22 | 25.80 | 45.12 | 11.49 | 12.01 | 30.91 | reject | reject | reject |
| 1.30 | 50.62 | 54.36 | 33.83 | 50.84 | 19.03 | 20.81 | 36.27 | reject | reject | reject |

## High-criticality streams

| U | round_robin | fifo | fixed_priority | edf | edf_batch | edf_batch_same_deadline | dynamic_batch | test: none | test: same_deadline | test: cross_deadline |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|
| 0.30 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | admit | admit | admit |
| 0.40 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | admit | admit | admit |
| 0.50 | 0.09 | 0.09 | 0.00 | 0.00 | 0.00 | 0.00 | 0.17 | admit | admit | admit |
| 0.60 | 4.00 | 4.00 | 0.01 | 0.01 | 0.00 | 0.00 | 4.91 | admit | admit | admit |
| 0.70 | 8.33 | 8.33 | 0.05 | 0.05 | 0.03 | 0.04 | 8.72 | admit | admit | reject |
| 0.80 | 8.85 | 8.83 | 0.61 | 0.61 | 0.04 | 0.04 | 8.88 | reject | admit | reject |
| 0.90 | 9.24 | 9.31 | 6.33 | 6.32 | 0.20 | 0.24 | 9.64 | reject | reject | reject |
| 1.00 | 18.20 | 15.91 | 13.38 | 14.16 | 0.66 | 0.64 | 10.33 | reject | reject | reject |
| 1.10 | 27.24 | 27.57 | 23.49 | 31.71 | 6.91 | 7.83 | 16.64 | reject | reject | reject |
| 1.20 | 44.83 | 42.12 | 24.53 | 43.95 | 11.74 | 12.27 | 31.58 | reject | reject | reject |
| 1.30 | 51.72 | 55.54 | 32.43 | 49.78 | 19.44 | 21.27 | 37.06 | reject | reject | reject |
