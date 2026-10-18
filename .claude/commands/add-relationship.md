Add a new relationship type to the atlas: $ARGUMENTS

1. Describe in one sentence what it finds and why a researcher would care; add it to the
   docstring of `src/atlas/analysis.py` and the table in README.
2. Plant a positive example (and a near-miss) in `tests/conftest.py::collection`, then
   write the failing test in `tests/test_build.py`.
3. Implement it in `analysis.relationships`, with thresholds as keyword arguments and an
   `evidence` string a non-programmer can read.
4. Make sure existing planted cases are still found and nothing new is flagged by accident.
5. Run `pytest -q`, `ruff check .`, and `atlas build --embedder hash --no-images`.
