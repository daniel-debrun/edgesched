# Contributing

Issues and pull requests are welcome. Measurements from real edge devices
(Jetson Orin, other embedded GPUs/NPUs) are especially useful.

## Development setup

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'        # add torch separately if you want the torch tests
ruff check . && ruff format --check src tests
pytest
```

## Guidelines

- Policies must stay pure: no clocks, threads or backend calls inside `choose()`.
  That is what lets the simulator and the runtime share them.
- A new policy needs hand-constructed queue tests and, if it batches, a check of
  the invariant it claims (see `tests/test_policies.py`).
- Changes to `edgesched.analysis` should cite the result they implement and keep
  `test_admitted_sets_have_no_misses_in_simulation` passing.
- Any number added to the README must come with the command that reproduces it
  and must say where it was measured (simulator, CPU, or which device).
- Keep the test suite under two minutes.
