import pytest

from research.floodwarp_methods_paper import runner
from research.floodwarp_methods_paper.fidelity.verification import suite_sources, verify_pass


def test_verification_resume_authenticates_test_log_summary_and_sources(tmp_path):
    source = {"tests/regression.py": "code"}
    env = {"CUDA_VISIBLE_DEVICES": ""}
    hw = {"gpu": "GPU", "python": "3.14", "cpu_model": "CPU", "warp": "1.17", "numpy": "n"}
    log = tmp_path / "attempt_1.log"
    log.write_text("fixture\n3 passed in 0.12s\n")
    path = tmp_path / "passed.json"
    saved = dict(source=source, environment=env, hardware=hw, command=["python", "-m", "pytest", "-q"],
                 returncode=0, log=log.name, sha256=runner.checksum(path=log), summary="3 passed in 0.12s")
    runner.write_json(path=path, data=saved)
    verify_pass(path=path, source=source, environment=env, hardware=hw)
    with pytest.raises(ValueError, match="source, environment or test log"):
        verify_pass(path=path, source={"tests/regression.py": "changed"}, environment=env, hardware=hw)
    runner.write_json(path=path, data=dict(saved, summary="4 passed in 0.12s"))
    with pytest.raises(ValueError, match="summary"):
        verify_pass(path=path, source=source, environment=env, hardware=hw)
    runner.write_json(path=path, data=saved)
    log.write_text("replaced\n3 passed in 0.12s\n")
    with pytest.raises(ValueError, match="test log"):
        verify_pass(path=path, source=source, environment=env, hardware=hw)


def test_regression_suite_identity_includes_research_code_configuration_and_pins(tmp_path):
    study = tmp_path / "research/paper"
    directory = study / "fidelity"
    configs = study / "configs/storms/test.json"
    code = study / "figures/native_extent.py"
    for path in (tmp_path/"pyproject.toml", study/"supplementary/constraints.txt", configs, code,
                 tmp_path/"validation/analytical/reference.py"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original")
    original = suite_sources(root=tmp_path, directory=directory)
    for path in (configs, code):
        path.write_text("modified")
        new = suite_sources(root=tmp_path, directory=directory)
        assert original[str(path.relative_to(tmp_path))] != new[str(path.relative_to(tmp_path))]
