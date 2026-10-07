def test_stable_public_import_is_available():
    from pathlib import Path

    import ye3t_methods
    import ye3t_methods.atomistic

    assert ye3t_methods.YE3TDescriptors is not None
    assert Path(ye3t_methods.atomistic.__file__).resolve().parents[2] == (
        Path(ye3t_methods.__file__).resolve().parents[1]
    )
    package = Path(ye3t_methods.atomistic.__file__).parent
    assert not (package / "ml").exists()
    assert not (package / "nn").exists()
    assert not (package / "phi_graph.py").exists()


def test_public_import_is_independent_of_a_preloaded_old_provider(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    old_package = tmp_path / "ye3t_ace"
    old_package.mkdir()
    (old_package / "__init__.py").write_text("OLD_PROVIDER = True\n")
    current_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((
        str(tmp_path), str(current_root), environment.get("PYTHONPATH", ""),
    ))
    preloaded = subprocess.run(
        [sys.executable, "-c", (
            "import ye3t_ace, ye3t_methods; "
            "print(ye3t_methods.Basis.__module__)"
        )],
        cwd=tmp_path, env=environment, capture_output=True, text=True,
    )
    assert preloaded.returncode == 0, preloaded.stderr
    assert preloaded.stdout.strip() == "ye3t_methods.linear"
    reader = subprocess.run(
        [sys.executable, "-c", (
            "from ye3t_methods._saved_model_compat import ensure_saved_model_imports; "
            "ensure_saved_model_imports(); import ye3t_ace; print(ye3t_ace.__file__)"
        )],
        cwd=tmp_path, env=environment, capture_output=True, text=True,
    )
    assert reader.returncode == 0, reader.stderr
    assert Path(reader.stdout.strip()).resolve() == (
        current_root / "ye3t_ace" / "__init__.py"
    )


def test_saved_model_shim_resolves_to_current_implementation(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    current_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((
        str(current_root), environment.get("PYTHONPATH", ""),
    ))
    result = subprocess.run(
        [sys.executable, "-c", (
            "import ye3t_ace; "
            "from ye3t_ace.ace.linear_ace import LinearACEScalarModelBundle as old; "
            "from ye3t_methods.atomistic.ace.linear_ace import LinearACEScalarModelBundle as new; "
            "print(ye3t_ace.__file__); print(old is new)"
        )],
        cwd=tmp_path, env=environment, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    origin, identity = result.stdout.strip().splitlines()
    assert Path(origin).resolve() == current_root / "ye3t_ace" / "__init__.py"
    assert identity == "True"


def test_saved_model_reader_rejects_preloaded_foreign_provider(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    old_package = tmp_path / "ye3t_ace"
    old_package.mkdir()
    (old_package / "__init__.py").write_text("OLD_PROVIDER = True\n")
    current_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(tmp_path), str(current_root)))
    result = subprocess.run(
        [sys.executable, "-c", (
            "import ye3t_ace; "
            "from ye3t_methods._saved_model_compat import ensure_saved_model_imports; "
            "ensure_saved_model_imports()"
        )],
        cwd=tmp_path, env=environment, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "A different ye3t_ace package was already imported" in result.stderr


def test_base_import_does_not_require_optional_extras(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    current_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(current_root),
                                                  environment.get("PYTHONPATH", "")))
    program = (
        "import sys\n"
        "class BlockExtras:\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname.split('.', 1)[0] in "
        "{'sklearn', 'matscipy', 'matplotlib', 'psutil'}:\n"
        "            raise ModuleNotFoundError(fullname)\n"
        "sys.meta_path.insert(0, BlockExtras())\n"
        "from ye3t_methods import Basis, LinearModel\n"
        "print(Basis.__name__, LinearModel.__name__)\n"
    )
    result = subprocess.run([sys.executable, "-c", program], cwd=tmp_path,
                            env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "Basis LinearModel"


def test_maintained_modules_have_no_historical_imports():
    import ast
    from pathlib import Path

    package = Path(__file__).resolve().parents[1] / "ye3t_methods"
    historical = []
    for path in package.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            if isinstance(node, ast.ImportFrom) and (
                node.module or ""
            ).startswith("ye3t_ace"):
                historical.append((path, node.lineno))
            if isinstance(node, ast.Import):
                historical.extend(
                    (path, node.lineno) for alias in node.names
                    if alias.name.startswith("ye3t_ace")
                )
    assert not historical
