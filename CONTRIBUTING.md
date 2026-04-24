# Contributing to longcombat-py

Thanks for your interest. This doc covers contributions to the **Python
port**. For the original R package, see
[`jcbeer/longCombat`](https://github.com/jcbeer/longCombat).

## Ways to contribute

- **Report a bug.** Open an issue with a minimal reproduction: the
  dataset shape, the `long_combat(...)` call you made, and what you
  expected vs. what you got.
- **Fix a bug or small behavior gap with R.** A failing test showing the
  discrepancy against the fixtures in `tests/fixtures/` is the clearest
  starting point.
- **Add a missing feature from the R package.** Note it in the issue
  first so we can agree on API naming (see "neuroCombat parity" below).
- **Improve docs or type hints.** Always welcome.

## Dev setup

```bash
git clone https://github.com/NoiseFilterT/longcombat-py.git
cd longcombat-py
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
pytest
```

To run the R-equivalence tests (optional — the CSV fixtures are
committed so you don't need R to run them):

```bash
pytest tests/test_r_equivalence.py
```

To regenerate the fixtures (requires R + `lme4` + `longCombat`), see
[`tests/fixtures/README.md`](tests/fixtures/README.md).

## Coding conventions

- **Type hints everywhere.** Public API functions have NumPy-style
  docstrings.
- **Match `neuroCombat` parameter names** where the concept overlaps
  (`batch_col`, `eb`, `mean_only`). Invent a new name only when
  `neuroCombat` has no equivalent, and flag it in the PR.
- **Preserve the math of the R original.** Any change that moves
  Python output away from R output must be justified in the PR
  description and (if intentional) documented in
  [`DIFFERENCES_FROM_R.md`](DIFFERENCES_FROM_R.md).
- **No new runtime dependencies** without discussion. `numpy`, `scipy`,
  `pandas`, `statsmodels` only. `matplotlib` is an optional extra.

## Commit messages

We use [Conventional Commits](https://www.conventionalcommits.org/) so
that versioning and changelogs can be automated with
[commitizen](https://commitizen-tools.github.io/commitizen/). Common
types:

| Type | Use for |
|---|---|
| `feat:` | new user-visible behavior |
| `fix:` | bug fixes |
| `docs:` | README / docstring / docs changes |
| `test:` | tests only |
| `refactor:` | internal refactors with no behavior change |
| `chore:` | build, deps, tooling |
| `perf:` | performance improvements |

Breaking changes get an `!` (`feat!: rename batch_col to batch`) or a
`BREAKING CHANGE:` footer.

## Running tests

```bash
pytest                              # full suite
pytest -k long_combat               # single test group
pytest tests/test_r_equivalence.py  # R-equivalence only
```

The test suite is fast (<5s locally). CI runs it on Python 3.10-3.13.

## Releases (maintainers)

1. Land changes on `main` via PRs with Conventional Commit titles.
2. Run `cz bump --yes` to generate a bump commit + version tag.
3. `git push origin main --follow-tags`.

## License

By contributing you agree that your contributions are licensed under the
Artistic License 2.0 (see [`LICENSE`](LICENSE)), the same license under
which this package is distributed.
