# GitHub Public Setup Guide

This document records the public visibility transition for the Noctilux repository.

Status: **Completed** (v0.3.5). The repository is now public.

## Repository Description (applied)

```
Offline image batch processing and augmentation toolkit — YAML-driven, reproducible, metadata-traceable pipelines for pre-training data preparation.
```

## Topics (applied)

`image-processing`, `data-augmentation`, `computer-vision`, `python`, `pillow`, `numpy`, `cli`, `dataset-tools`, `offline-processing`, `yaml-config`

```
Offline image batch processing and augmentation toolkit — YAML-driven, reproducible, metadata-traceable pipelines for pre-training data preparation.
```

## Topics (applied)

`image-processing`, `data-augmentation`, `computer-vision`, `python`, `pillow`, `numpy`, `cli`, `dataset-tools`, `offline-processing`, `yaml-config`

## Pre-Public Checklist (completed)

Before changing the repository from private to public:

1. Complete all items in `docs/public_readiness.md`.
2. Verify no secrets, tokens, or personal paths remain in committed files.
3. Verify `examples/images/sample.jpg` is a safe synthetic image.
4. Verify CI passes on the latest `main` commit and latest tag.
5. Verify `outputs/` is not tracked by git.
6. Confirm the `v0.3.0` tag has not been moved.

## Visibility Change (completed)

These steps must be performed manually by the repository owner. Agents must not automate this change.

1. Go to **Settings > General > Danger Zone** in the GitHub repository.
2. Click **Change repository visibility**.
3. Select **Make public**.
4. Confirm the action.

## Post-Public Status

- CI badge renders correctly in `README.md`.
- All documentation links are accessible.
- Repository description and topics are set.

- Verify the CI badge renders correctly in `README.md`.
- Verify all documentation links are accessible.
- Consider adding a `.github/ISSUE_TEMPLATE/` and `.github/PULL_REQUEST_TEMPLATE.md` for community contributions.
- Monitor the first few CI runs after visibility change for any permission issues.

## What Not to Do Now

- Do not publish to PyPI without explicit approval.
- Do not create GitHub Releases unless release notes are ready.
- Do not move, delete, or force-push any existing tags.
