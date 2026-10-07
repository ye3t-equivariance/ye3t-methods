"""Explicit catalogue prewarm must stay bounded and preserve compiler labels."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from ye3t.couplings import count
from ye3t_methods.cache import prewarm_descriptor_catalogue


def _catalogue():
    label = count(
        content=(1, 1), input_Ls=(0, 0), target_L=0,
        target_permutation="trivial", carrier="ACE_density",
        tree_schedule="balanced", validation_scope="counts",
    ).labels_for_target(0)[0]
    return {
        "schema": "ye3t_methods_prewarm_v1",
        "settings": {
            "ranks": [2], "basis_type": "no_charge", "elems": ["Cu"],
            "nmax": [1], "lmax": [0], "lmin": [0],
            "L_R": 0, "M_R_values": [0],
        },
        "selected_labels": [label.to_dict()],
    }


def test_preview_and_budget_refusal_do_not_compile_or_write(tmp_path, monkeypatch):
    import ye3t_methods.cache as prewarm_module

    catalogue = _catalogue()
    def forbidden(*args, **kwargs):
        raise AssertionError("Coefficient compiler ran during preview or refusal.")

    monkeypatch.setattr(prewarm_module, "compile_descriptor_artifacts", forbidden)
    cache_dir = tmp_path / "uncreated_cache"
    preview = prewarm_descriptor_catalogue(catalogue, cache_dir=cache_dir)
    assert preview["selected_labels"] == 1
    assert preview["schema"] == "ye3t_methods_prewarm_report_v1"
    assert preview["magnetic_input_elements_proxy"] == 1
    assert preview["applied"] is False
    assert not cache_dir.exists()
    with monkeypatch.context() as patch:
        patch.setattr(prewarm_module, "count", forbidden)
        with pytest.raises(ValueError, match="max_labels"):
            prewarm_descriptor_catalogue(
                {**catalogue, "selected_labels": catalogue["selected_labels"] * 2},
                cache_dir=cache_dir, max_labels=1, apply=True,
            )
        with pytest.raises(ValueError, match="max_rank"):
            prewarm_descriptor_catalogue(catalogue, cache_dir=cache_dir,
                                         max_rank=1, apply=True)
    angular = count(
        content=(1, 1), input_Ls=(1, 1), target_L=0,
        target_permutation="trivial", carrier="ACE_density",
        tree_schedule="balanced", validation_scope="counts",
    ).labels_for_target(0)[0]
    expensive = json.loads(json.dumps(catalogue))
    expensive["settings"]["lmax"] = [1]
    expensive["selected_labels"] = [angular.to_dict()]
    with monkeypatch.context() as patch:
        patch.setattr(prewarm_module, "count", forbidden)
        with pytest.raises(ValueError, match="max_magnetic_elements"):
            prewarm_descriptor_catalogue(expensive, cache_dir=cache_dir,
                                         max_magnetic_elements=1, apply=True)
    assert not cache_dir.exists()


def test_old_catalogue_schema_still_reads_without_compilation(tmp_path):
    catalogue = _catalogue()
    catalogue["schema"] = "ye3t_ace_prewarm_v1"
    report = prewarm_descriptor_catalogue(catalogue, cache_dir=tmp_path / "cache")
    assert report["schema"] == "ye3t_ace_prewarm_report_v1"
    assert not (tmp_path / "cache").exists()


def test_prewarm_replays_selected_artifacts_without_coefficient_recompile(
    tmp_path, monkeypatch,
):
    from ye3t_methods.atomistic.equivariant_calc import descriptor_sets

    catalogue = _catalogue()
    cache_dir = tmp_path / "cache"
    first = prewarm_descriptor_catalogue(catalogue, cache_dir=cache_dir,
                                         apply=True)
    assert first["applied"] is True
    assert list(cache_dir.glob("artifacts/ace_descriptor_artifacts/*.json"))
    assert list(cache_dir.glob("arrays/*.npz"))

    def forbidden(*args, **kwargs):
        raise AssertionError("Warm prewarm recompiled coefficient data.")

    monkeypatch.setattr(descriptor_sets, "_compiled_scalar_coordinate_library", forbidden)
    second = prewarm_descriptor_catalogue(catalogue, cache_dir=cache_dir,
                                          apply=True)
    assert second["cache_stats"]["artifact_disk_hits"] == 1

    monkeypatch.setenv("YE3T_CACHE_MODE", "read_only")
    with pytest.raises(ValueError, match="writable cache mode"):
        prewarm_descriptor_catalogue(catalogue, cache_dir=cache_dir, apply=True)
    monkeypatch.delenv("YE3T_CACHE_MODE")

    invalid = json.loads(json.dumps(catalogue))
    invalid["selected_labels"][0]["basis_key"] = ["sym", 0, 99]
    with pytest.raises(ValueError, match="not issued"):
        prewarm_descriptor_catalogue(invalid, cache_dir=cache_dir, apply=True)


def test_prewarm_cli_preview_and_apply(tmp_path):
    path = tmp_path / "catalogue.json"
    path.write_text(json.dumps(_catalogue()), encoding="utf-8")
    cache_dir = tmp_path / "cache"
    command = [sys.executable, "-m", "ye3t_methods.cache", "--catalogue",
               str(path), "--cache-dir", str(cache_dir)]
    preview = subprocess.run(command, check=True, capture_output=True, text=True)
    assert json.loads(preview.stdout)["applied"] is False
    assert not cache_dir.exists()
    applied = subprocess.run(command + ["--apply"], check=True,
                             capture_output=True, text=True)
    assert json.loads(applied.stdout)["applied"] is True
    assert list(Path(cache_dir).glob("artifacts/ace_descriptor_artifacts/*.json"))
