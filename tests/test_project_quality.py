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
        Path("configs/examples/opencv_backend.yaml"),
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


def test_opencv_backend_module_exists() -> None:
    assert Path("src/noctilux/backends/__init__.py").exists()
    assert Path("src/noctilux/backends/opencv_backend.py").exists()


def test_opencv_config_example_exists() -> None:
    assert Path("configs/examples/opencv_backend.yaml").exists()


def test_opencv_config_validates() -> None:
    config = resolve_config(load_config(Path("configs/examples/opencv_backend.yaml")))
    validate_config(config)


def test_opencv_optional_dependency_in_pyproject() -> None:
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    assert "opencv" in text
    assert "opencv-python-headless" in text


def test_ci_workflow_has_opencv_backend_job() -> None:
    text = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "opencv-backend:" in text
    assert "test_opencv_backend.py" in text
    assert "opencv_backend.yaml" in text


def test_ci_opencv_backend_job_installs_opencv_extra() -> None:
    text = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert ".[dev,opencv]" in text


def test_ci_uses_modern_actions() -> None:
    text = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "actions/checkout@v6" in text
    assert "actions/setup-python@v6" in text
    assert "actions/checkout@v4" not in text
    assert "actions/setup-python@v5" not in text


def test_opencv_missing_backend_error_message() -> None:
    import sys
    from unittest.mock import patch

    from noctilux.backends.opencv_backend import require_opencv

    cv2_mod = sys.modules.pop("cv2", None)
    try:
        with patch("noctilux.backends.opencv_backend.is_opencv_available", return_value=False):
            with __import__("pytest").raises(Exception) as exc_info:
                require_opencv()
            msg = str(exc_info.value)
            assert "noctilux[opencv]" in msg
    finally:
        if cv2_mod is not None:
            sys.modules["cv2"] = cv2_mod


def test_parallel_resume_design_doc_exists() -> None:
    assert Path("docs/parallel_resume_design.md").exists()


def test_parallel_resume_design_mentions_metadata_safe_writer() -> None:
    text = Path("docs/parallel_resume_design.md").read_text(encoding="utf-8")
    assert "metadata-safe" in text.lower() or "MetadataWriter" in text


def test_parallel_resume_design_mentions_deterministic_seed() -> None:
    text = Path("docs/parallel_resume_design.md").read_text(encoding="utf-8")
    assert "deterministic" in text.lower() or "combine_seed" in text


def test_parallel_resume_design_mentions_resume_flags() -> None:
    text = Path("docs/parallel_resume_design.md").read_text(encoding="utf-8")
    assert "--resume" in text
    assert "--skip-existing" in text
    assert "--retry-failed" in text


def test_project_guide_mentions_v05x_parallel_resume() -> None:
    text = Path("PROJECT_GUIDE.md").read_text(encoding="utf-8")
    v050_section = text[text.find("v0.5.0") : text.find("v0.6.0")] if "v0.5.0" in text else ""
    assert v050_section, "PROJECT_GUIDE.md should have a v0.5.0 section"
    assert "resume" in v050_section.lower() or "parallel" in v050_section.lower()


def test_readme_mentions_parallel_resume_roadmap() -> None:
    text = Path("README.md").read_text(encoding="utf-8")
    assert "parallel" in text.lower()
    assert "resume" in text.lower()


# --- v0.7.0 annotation sync design tests ---


def test_annotation_sync_design_doc_exists() -> None:
    assert Path("docs/annotation_sync_design.md").exists()


def test_readme_mentions_annotation_sync_roadmap_but_not_supported() -> None:
    text = Path("README.md").read_text(encoding="utf-8")
    assert "annotation" in text.lower()
    assert "not yet supported" in text.lower() or "not yet" in text.lower()


def test_project_guide_mentions_v07x_annotation_sync() -> None:
    text = Path("PROJECT_GUIDE.md").read_text(encoding="utf-8")
    end_marker = "v0.8.0" if "v0.8.0" in text else "v0.8.1"
    v07_section = text[text.find("v0.7") : text.find(end_marker)] if "v0.7" in text else ""
    assert v07_section, "PROJECT_GUIDE.md should have a v0.7.x section"
    assert "annotation" in v07_section.lower()


def test_agent_handoff_mentions_annotation_sync_design_only() -> None:
    text = Path("docs/agent_handoff.md").read_text(encoding="utf-8")
    assert "annotation" in text.lower()
    assert "design" in text.lower()


def test_annotation_sync_design_mentions_formats_and_types() -> None:
    text = Path("docs/annotation_sync_design.md").read_text(encoding="utf-8")
    assert "COCO" in text
    assert "YOLO" in text
    assert "bbox" in text.lower() or "bounding box" in text.lower()
    assert "mask" in text.lower()
    assert "keypoint" in text.lower()


def test_annotation_sync_design_confirms_image_only_unchanged() -> None:
    text = Path("docs/annotation_sync_design.md").read_text(encoding="utf-8")
    assert "image-only" in text.lower() or "unchanged" in text.lower()


# --- v0.7.4 annotation writer tests ---


def test_annotation_writers_module_exists() -> None:
    assert Path("src/noctilux/annotations/writers.py").exists()


def test_annotation_sync_design_mentions_writers() -> None:
    text = Path("docs/annotation_sync_design.md").read_text(encoding="utf-8")
    assert "writer" in text.lower()


def test_agent_handoff_mentions_writer_prototype() -> None:
    text = Path("docs/agent_handoff.md").read_text(encoding="utf-8")
    assert "writer" in text.lower()
    assert "v0.7.4" in text


# --- v0.7.5 annotation writer cleanup tests ---


def test_agent_handoff_mentions_v075() -> None:
    text = Path("docs/agent_handoff.md").read_text(encoding="utf-8")
    assert "v0.7.5" in text


def test_annotation_sync_design_mentions_unique_ids() -> None:
    text = Path("docs/annotation_sync_design.md").read_text(encoding="utf-8")
    assert "unique" in text.lower()


def test_changelog_has_v075() -> None:
    text = Path("CHANGELOG.md").read_text(encoding="utf-8")
    assert "0.7.5" in text

