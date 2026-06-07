from pathlib import Path
import json
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST_PATH = REPO_ROOT / ".ci" / "artifact-hygiene-allowlist.txt"
IGNORED_DIRS = {".git", ".pytest_cache"}
ARTIFACT_DIR_NAMES = {"__pycache__", "logs", "checkpoints"}
ARTIFACT_SUFFIXES = {".pyc", ".pyo"}
WORKFLOW_CONFIG_DIR = REPO_ROOT / "configs" / "pretraining"
WORKFLOW_DOC = REPO_ROOT / "docs" / "pretraining_workflows.md"


def project_python_files():
    for path in sorted(REPO_ROOT.rglob("*.py")):
        if IGNORED_DIRS.intersection(path.parts):
            continue
        yield path


def read_allowlist():
    entries = []
    for line in ALLOWLIST_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            entries.append(stripped)
    return entries


def tracked_files():
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return set(result.stdout.splitlines())


def is_generated_artifact(path):
    parts = Path(path).parts
    return bool(ARTIFACT_DIR_NAMES.intersection(parts)) or Path(path).suffix in ARTIFACT_SUFFIXES


def test_python_sources_compile_without_importing_heavy_ml_dependencies():
    failures = []
    for path in project_python_files():
        source = path.read_text(encoding="utf-8")
        try:
            compile(source, str(path.relative_to(REPO_ROOT)), "exec")
        except SyntaxError as exc:
            failures.append(f"{path.relative_to(REPO_ROOT)}: {exc}")

    assert not failures, "Python source compile failures:\n" + "\n".join(failures)


def test_no_new_generated_artifacts_are_tracked():
    allowlist = read_allowlist()
    assert allowlist == sorted(set(allowlist)), "Artifact hygiene allowlist must be sorted and unique."

    tracked = tracked_files()
    generated_artifacts = sorted(path for path in tracked if is_generated_artifact(path))
    unexpected = sorted(set(generated_artifacts) - set(allowlist))
    stale = sorted(set(allowlist) - tracked)

    assert not unexpected, "New tracked generated artifacts are not allowed:\n" + "\n".join(unexpected)
    assert not stale, "Remove stale entries from .ci/artifact-hygiene-allowlist.txt:\n" + "\n".join(stale)


def test_pretraining_workflow_configs_are_parseable_and_issue_linked():
    expected = {
        "chemberta2_mlm.json": 3,
        "property_prediction_multitask.json": 5,
        "molecule_generation_vae.json": 6,
    }

    for file_name, issue_number in expected.items():
        config_path = WORKFLOW_CONFIG_DIR / file_name
        config = json.loads(config_path.read_text(encoding="utf-8"))
        assert config["issue"] == issue_number
        assert config["workflow"]
        assert "data" in config


def test_pretraining_workflow_docs_cover_open_issue_workflows():
    body = WORKFLOW_DOC.read_text(encoding="utf-8").lower()
    required_markers = [
        "issue #3",
        "issue #5",
        "issue #6",
        "chemberta2",
        "property prediction",
        "molecular generation",
        "random-weight",
    ]

    missing = [marker for marker in required_markers if marker not in body]
    assert not missing, "Workflow documentation is missing markers: " + ", ".join(missing)
