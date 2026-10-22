"""Keep example outputs inside the test's temporary workflow directory."""
import pytest


@pytest.fixture(autouse=True)
def example_workflow_directory(tmp_path, monkeypatch):
    monkeypatch.setenv('YE3T_WORKFLOW_ROOT', str(tmp_path / 'ye3t-workflows'))
