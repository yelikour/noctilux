# Public Readiness Checklist

This document tracks what needs to be verified before making the Noctilux repository public.

Use this checklist before switching the repository from private to public. See also `docs/github_public_setup.md` for suggested metadata and manual steps.

Current status: The repository is now public (as of v0.3.5). This checklist was completed before the visibility change.

## Repository Basics

- [ ] `LICENSE` file exists and contains the correct MIT license text
- [ ] `README.md` has project description, installation, quickstart, and usage examples
- [ ] `CHANGELOG.md` is up to date with the latest release
- [ ] `pyproject.toml` version matches the latest tag
- [ ] `src/noctilux/__init__.py` version matches `pyproject.toml`

## Documentation

- [ ] `docs/agent_handoff.md` reflects current project state and version
- [ ] `docs/getting_started.md` is accurate
- [ ] `docs/configuration.md` is accurate
- [ ] `docs/input_formats.md` is accurate
- [ ] `docs/output_formats.md` is accurate
- [ ] `docs/adding_new_transform.md` is accurate
- [ ] `docs/github_public_setup.md` exists and provides public setup guidance
- [ ] All example configs in `configs/examples/` are runnable
- [ ] All preset configs in `configs/presets/` are valid

## CI and Quality

- [ ] GitHub Actions CI passes on `main`
- [ ] `python -m pytest` passes
- [ ] `ruff check src tests scripts` passes
- [ ] `noctilux --help` works
- [ ] `noctilux list-transforms` lists all expected transforms
- [ ] `noctilux preview --help` works
- [ ] `noctilux report --help` works
- [ ] `noctilux inspect-config --config configs/examples/quickstart_sample.yaml` works
- [ ] Latest tag CI passes

## Python Version Support

- [ ] `pyproject.toml` classifiers only list Python versions covered by CI
- [ ] CI matrix matches declared support range (currently 3.10, 3.11, 3.12)
- [ ] `requires-python` is consistent with CI coverage

## Sample Assets

- [ ] `examples/images/sample.jpg` exists, is synthetic, and has no privacy or copyright issues
- [ ] `examples/images/sample.jpg` is under 1 MB
- [ ] Sample image is tracked in git
- [ ] Sample image is not a real photograph with identifiable subjects

## Cleanliness

- [ ] No `outputs/` directory committed (confirmed via `.gitignore`)
- [ ] No `__pycache__` / `.pytest_cache` / `.ruff_cache` committed
- [ ] No virtual environment directories committed
- [ ] No large binary files committed (except `examples/images/sample.jpg`)
- [ ] No hardcoded absolute paths in committed files
- [ ] No secrets, tokens, API keys, or credentials in committed files
- [ ] `.gitignore` covers common artifact directories

## GitHub Metadata

- [ ] Repository description is set (see `docs/github_public_setup.md` for suggestions)
- [ ] Repository topics are set (see `docs/github_public_setup.md` for suggestions)
- [ ] Default branch is `main`
- [ ] No open draft PRs with sensitive content

## Tag and Release State

- [ ] Current version tag points to the correct commit
- [ ] `v0.3.0` tag is unchanged (known CI failure is documented history; do not move this tag)
- [ ] The latest version is the recommended starting point for new users

## Sensitive Information

- [ ] No `ghp_` tokens in committed files
- [ ] No `BEGIN OPENSSH PRIVATE KEY` blocks in committed files
- [ ] No `api_key=` assignments with real values in committed files
- [ ] No hardcoded `/home/username` paths in documentation or source

## Current Publication Scope

- The repository is now public.
- Community health files are in place: issue templates, PR template, SECURITY.md, CODE_OF_CONDUCT.md, CONTRIBUTING.md.
- Noctilux does not publish to PyPI yet.
- Distribution is via `git clone` and `pip install -e .`.

## Known Historical Notes

- `v0.3.0` tag points to `ea4905a` and has a known CI failure due to a pre-fix commit. The tag must not be moved, deleted, or force-pushed. `v0.3.1` and later are the corrected releases.
