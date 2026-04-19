# Contributing

Thanks for considering a contribution. This project scores manipulation risk on
Polymarket markets; correctness matters, because users make position-sizing
decisions from the output.

## Getting set up

```bash
git clone https://github.com/ksk5429/polymarket-oracle-risk
cd polymarket-oracle-risk
uv sync --all-extras
uv run pre-commit install
```

## Development loop

```bash
uv run pytest               # fast suite
uv run pytest -m slow       # full suite including backtest
uv run pyright src tests
uv run ruff check --fix .
uv run ruff format .
```

## Pull-request checklist

- [ ] Conventional-commit title (`feat:`, `fix:`, `docs:`, …)
- [ ] New public API has a type hint and a docstring
- [ ] Backtest Brier/ECE unchanged or improved (report in PR body)
- [ ] Zelenskyy-suit regression test still scores ≥ 0.75 subjectivity
- [ ] No new runtime dependency without a justification
- [ ] `CHANGELOG.md` updated under `## [Unreleased]`

## Reproducing the backtest

Historical features are re-fetched from Goldsky subgraphs and Polymarket's Gamma
API. See `docs/backtest.md` for the reproducible recipe.
