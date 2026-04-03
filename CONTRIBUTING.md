# Contributing to Dystrio Sculpt

We welcome contributions. Here's how to get started.

## Setup

```bash
git clone https://github.com/dystrio/sculpt.git
cd sculpt
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/
```

Some tests are marked `@pytest.mark.slow` and require GPU + model downloads. Skip them with:

```bash
pytest tests/ -m "not slow"
```

## Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
ruff check src/ tests/
ruff format src/ tests/
```

## Adding a New Model Architecture

1. Check if your model's `model_type` is already in `src/dystrio_sculpt/architectures/fingerprint.py`
2. If not, add it to `_KNOWN_ARCHITECTURES` with the correct MLP type and family
3. If the FFN structure is non-standard, create a new adapter in `src/dystrio_sculpt/architectures/`
4. Register the adapter in `src/dystrio_sculpt/architectures/__init__.py`
5. Add tests in `tests/`

See `swiglu_dense.py` (dense models) or `swiglu_moe.py` (MoE models) for reference adapters.

## Adding a New Workload

Add your workload to `MIXTURE_PRESETS` in `src/dystrio_sculpt/_data.py`. Each source needs:

- `dataset`: HuggingFace dataset ID
- `split`: dataset split
- `weight`: relative weight in the mixture
- `text_field` or `formatter`: how to extract text from each row

## Pull Requests

- Keep PRs focused on a single change
- Include tests for new functionality
- Run `ruff check` and `pytest` before submitting
- Describe what changed and why in the PR description

## Reporting Issues

Open a GitHub issue with:

- What you tried (command, model, hardware)
- What happened (error message, log output)
- What you expected

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
