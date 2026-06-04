from __future__ import annotations

from pathlib import Path

from noctilux.config import load_config, resolve_config, validate_config
from noctilux.registry import list_transforms


def test_all_example_and_preset_configs_validate() -> None:
    config_paths = sorted(Path("configs/examples").glob("*.yaml")) + sorted(Path("configs/presets").glob("*.yaml"))
    assert config_paths

    for config_path in config_paths:
        config = resolve_config(load_config(config_path))
        validate_config(config)


def test_readme_key_config_paths_exist() -> None:
    paths = [
        Path("configs/examples/full_v020.yaml"),
        Path("configs/examples/quickstart_sample.yaml"),
        Path("configs/presets/all_basic_v021.yaml"),
        Path("examples/images/sample.jpg"),
        Path("configs/presets/classification_light.yaml"),
        Path("configs/presets/compression_robustness.yaml"),
        Path("scripts/preview_transforms.py"),
    ]
    for path in paths:
        assert path.exists(), path


def test_transform_registry_count_has_expected_floor() -> None:
    assert len(list_transforms()) >= 27


def test_preset_pipeline_names_are_safe_path_components() -> None:
    for config_path in sorted(Path("configs/presets").glob("*.yaml")):
        config = resolve_config(load_config(config_path))
        validate_config(config)
        for pipeline in config["pipelines"]:
            name = pipeline["name"]
            assert "/" not in name
            assert "\\" not in name
            assert name not in {".", ".."}


def test_ci_workflow_exists_and_runs_quality_checks() -> None:
    workflow_path = Path(".github/workflows/ci.yml")
    assert workflow_path.exists()
    text = workflow_path.read_text(encoding="utf-8")
    assert "python -m pytest" in text
    assert "ruff check src tests scripts" in text
    assert "noctilux preview --help" in text
    assert "noctilux preview" in text
    assert "examples/images/sample.jpg" in text
    assert "test -s /tmp/noctilux_ci_preview.jpg" in text
    assert "configs/presets/all_basic_v021.yaml" in text
    assert "configs/examples/quickstart_sample.yaml" in text
    assert "noctilux report" in text
    assert "test -s /tmp/noctilux_report.md" in text


def test_pyproject_classifiers_match_ci_python_versions() -> None:
    pyproject_path = Path("pyproject.toml")
    text = pyproject_path.read_text(encoding="utf-8")
    assert "Programming Language :: Python :: 3.10" in text
    assert "Programming Language :: Python :: 3.11" in text
    assert "Programming Language :: Python :: 3.12" in text
    assert "Programming Language :: Python :: 3.13" not in text


def test_ci_matrix_covers_supported_versions() -> None:
    ci_path = Path(".github/workflows/ci.yml")
    text = ci_path.read_text(encoding="utf-8")
    assert '"3.10"' in text
    assert '"3.11"' in text
    assert '"3.12"' in text


def test_public_readiness_doc_exists() -> None:
    assert Path("docs/public_readiness.md").exists()


def test_readme_mentions_public_readiness() -> None:
    text = Path("README.md").read_text(encoding="utf-8")
    assert "public_readiness.md" in text


def test_agent_handoff_mentions_public_readiness() -> None:
    text = Path("docs/agent_handoff.md").read_text(encoding="utf-8")
    assert "public_readiness.md" in text


def test_project_guide_roadmap_mentions_report() -> None:
    text = Path("PROJECT_GUIDE.md").read_text(encoding="utf-8")
    assert "report" in text.lower()
    v030_section = text[text.find("v0.3.x") : text.find("v0.4.0")] if "v0.3.x" in text else ""
    assert v030_section, "PROJECT_GUIDE.md should have a v0.3.x section"
    assert "report" in v030_section.lower()
