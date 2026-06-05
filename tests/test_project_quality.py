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


def test_github_public_setup_doc_exists() -> None:
    assert Path("docs/github_public_setup.md").exists()


def test_readme_mentions_core_commands() -> None:
    text = Path("README.md").read_text(encoding="utf-8")
    assert "noctilux run" in text
    assert "noctilux preview" in text
    assert "noctilux report" in text


def test_public_readiness_mentions_github_setup() -> None:
    text = Path("docs/public_readiness.md").read_text(encoding="utf-8")
    assert "github_public_setup.md" in text


def test_agent_handoff_mentions_v030_tag_immutable() -> None:
    text = Path("docs/agent_handoff.md").read_text(encoding="utf-8")
    assert "v0.3.0" in text
    assert "do not move" in text.lower() or "must not be moved" in text.lower() or "unchanged" in text.lower()


def test_agent_handoff_mentions_public_requires_manual_confirmation() -> None:
    text = Path("docs/agent_handoff.md").read_text(encoding="utf-8")
    assert "manually" in text.lower() or "manual" in text.lower()


def test_outputs_in_gitignore() -> None:
    text = Path(".gitignore").read_text(encoding="utf-8")
    assert "outputs" in text


def test_sample_image_tracked_and_small() -> None:
    import subprocess

    sample = Path("examples/images/sample.jpg")
    assert sample.exists()
    assert sample.stat().st_size < 1_000_000
    result = subprocess.run(["git", "ls-files", "--error-unmatch", str(sample)], capture_output=True, text=True)
    assert result.returncode == 0, f"sample.jpg not tracked by git: {result.stderr}"


def test_no_sensitive_patterns_in_docs() -> None:
    dangerous_patterns = ["ghp_", "BEGIN OPENSSH PRIVATE KEY", "api_key="]
    skip_files = {"docs/public_readiness.md"}
    doc_files = list(Path("docs").glob("*.md")) + [Path("README.md"), Path("CHANGELOG.md"), Path("pyproject.toml")]
    for path in doc_files:
        if str(path) in skip_files:
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in dangerous_patterns:
            assert pattern not in text, f"Found '{pattern}' in {path}"


def test_contributing_exists() -> None:
    assert Path("CONTRIBUTING.md").exists()


def test_readme_mentions_contributing() -> None:
    text = Path("README.md").read_text(encoding="utf-8")
    assert "CONTRIBUTING.md" in text


def test_readme_no_private_note() -> None:
    text = Path("README.md").read_text(encoding="utf-8")
    assert "currently private" not in text.lower()


def test_issue_templates_exist() -> None:
    assert Path(".github/ISSUE_TEMPLATE/bug_report.md").exists()
    assert Path(".github/ISSUE_TEMPLATE/feature_request.md").exists()


def test_pull_request_template_exists() -> None:
    assert Path(".github/pull_request_template.md").exists()


def test_security_md_exists() -> None:
    assert Path("SECURITY.md").exists()


def test_readme_mentions_roadmap() -> None:
    text = Path("README.md").read_text(encoding="utf-8")
    assert "PROJECT_GUIDE.md" in text or "roadmap" in text.lower()


def test_backend_design_doc_exists() -> None:
    assert Path("docs/backend_design.md").exists()


def test_readme_mentions_opencv_backend_roadmap() -> None:
    text = Path("README.md").read_text(encoding="utf-8")
    assert "OpenCV" in text


def test_project_guide_mentions_v040_backend() -> None:
    text = Path("PROJECT_GUIDE.md").read_text(encoding="utf-8")
    v040_section = text[text.find("v0.4.0") : text.find("v0.5.0")] if "v0.4.0" in text else ""
    assert v040_section, "PROJECT_GUIDE.md should have a v0.4.0 section"
    assert "backend" in v040_section.lower()


def test_backend_design_mentions_opencv_optional() -> None:
    text = Path("docs/backend_design.md").read_text(encoding="utf-8")
    assert "noctilux[opencv]" in text


def test_backend_design_confirms_pillow_default() -> None:
    text = Path("docs/backend_design.md").read_text(encoding="utf-8")
    assert "PIL.Image.Image" in text
    assert "default" in text.lower()

