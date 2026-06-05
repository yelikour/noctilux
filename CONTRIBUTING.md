# Contributing to Noctilux

Thank you for your interest in contributing to Noctilux.

## Setup

```bash
pip install -e ".[dev]"
```

## Development Workflow

1. Fork the repository and create a branch.
2. Make your changes.
3. Run quality checks:

```bash
python -m pytest
ruff check src tests scripts
```

4. Submit a pull request.

## Guidelines

- Run `pytest` and `ruff check src tests scripts` before every commit.
- New transforms must include unit tests and a YAML config example.
- Do not commit `outputs/`, caches, or virtual environments.
- Do not move, delete, or force-push any published git tags.
- Keep dependencies minimal. External backends (OpenCV, Albumentations, etc.) must be optional.
- Follow the existing code style. When in doubt, run `ruff`.

## Adding a New Transform

See [docs/adding_new_transform.md](docs/adding_new_transform.md) for a complete walkthrough.

## Reporting Issues

Open an issue at [github.com/yelikour/noctilux/issues](https://github.com/yelikour/noctilux/issues).

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
