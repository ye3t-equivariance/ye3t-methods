"""Verified JSON/NPZ persistence for configured density plus tagged scalar fits."""

from copy import deepcopy
from io import BytesIO
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np

from ye3t import YE3TRepresentation
from ye3t_methods.atomistic.ace.linear_ace import LinearACEScalarModelBundle
from ye3t_methods.atomistic.tagged_cauchy_image import load_tagged_cauchy_image_model


_SCHEMA = "ye3t_combined_scalar_bundle_v1"
_LIMIT = 256 * 1024 * 1024


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _pairs(rows):
    result = {}
    for key, value in rows:
        if key in result:
            raise ValueError("Combined bundle JSON contains a duplicate key.")
        result[key] = value
    return result


def _constant(value):
    raise ValueError("Combined bundle JSON contains a nonfinite constant: " + value)


def _finite_float(value):
    number = float(value)
    if not np.isfinite(number):
        raise ValueError("Combined bundle JSON contains a nonfinite number.")
    return number


def _json(data):
    return json.loads(data.decode("utf-8"), object_pairs_hook=_pairs,
                      parse_constant=_constant, parse_float=_finite_float)


def _npz_bytes(arrays):
    output = BytesIO()
    np.savez(output, **arrays)
    return output.getvalue()


def _npz(data, expected):
    with zipfile.ZipFile(BytesIO(data)) as archive:
        infos = archive.infolist()
        if (len(infos) != len(expected) or
                {item.filename for item in infos} != {name + ".npy" for name in expected} or
                any(item.file_size > 64 * 1024 * 1024 or item.flag_bits & 1
                    for item in infos) or
                sum(item.file_size for item in infos) > _LIMIT):
            raise ValueError("Combined bundle NPZ inventory is invalid.")
    with np.load(BytesIO(data), allow_pickle=False) as saved:
        result = {name: saved[name].copy() for name in expected}
    if any(array.dtype not in (np.dtype("float64"), np.dtype("int64")) or
           array.ndim > 2 or
           (array.dtype == np.dtype("float64") and not np.isfinite(array).all())
           for array in result.values()):
        raise ValueError("Combined bundle arrays must be bounded finite numeric data.")
    return result


def _array_inventory(arrays):
    return {name: {"shape": list(array.shape), "dtype": str(array.dtype)}
            for name, array in sorted(arrays.items())}


def _validate_posterior(posterior, weights, width):
    if (type(width) is not int or width < 0 or
            not isinstance(posterior, dict) or
            posterior.get("schema") != "ye3t_linear_ard_posterior_v1" or
            posterior.get("status") != "python_offline_only" or
            posterior.get("design_column_order") !=
            "component_features_then_species_offsets"):
        raise ValueError("Combined ARD posterior schema or column order is invalid.")
    active = posterior.get("active_column_indices")
    if (not isinstance(active, list) or
            any(type(index) is not int or index < 0 or index >= width
                for index in active) or len(set(active)) != len(active)):
        raise ValueError("Combined ARD posterior active columns are invalid.")
    precision = weights["posterior_precision"]
    covariance = weights["posterior_covariance"]
    threshold = posterior.get("threshold_lambda")
    noise = posterior.get("noise_precision")
    if (precision.shape != (width,) or covariance.shape != (len(active), len(active)) or
            len(active) > 2048 or
            type(threshold) not in (int, float) or not np.isfinite(threshold) or
            threshold <= 0 or type(noise) not in (int, float) or
            not np.isfinite(noise) or noise <= 0 or
            np.any(precision < 0) or
            active != np.flatnonzero(precision < threshold).tolist()):
        raise ValueError("Combined ARD posterior dimensions or precision are invalid.")
    if active:
        scale = max(1.0, float(np.max(np.abs(np.diag(covariance)))))
        tolerance = 1e-10 * scale
        if (not np.allclose(covariance, covariance.T, rtol=0, atol=tolerance) or
                np.any(np.diag(covariance) < -tolerance)):
            raise ValueError("Combined ARD posterior covariance is not symmetric.")
        symmetric = (covariance + covariance.T) / 2
        try:
            np.linalg.cholesky(symmetric + tolerance * np.eye(len(active)))
        except np.linalg.LinAlgError as error:
            raise ValueError("Combined ARD posterior covariance is not positive semidefinite.") from error


def _verified_members(raw):
    if len(raw) > _LIMIT:
        raise ValueError("Combined bundle exceeds the 256 MiB file limit.")
    with zipfile.ZipFile(BytesIO(raw)) as archive:
        infos = archive.infolist()
        names = [item.filename for item in infos]
        if "manifest.json" not in names:
            return None
        if archive.getinfo("manifest.json").file_size > 2 * 1024 * 1024:
            raise ValueError("Combined bundle manifest exceeds 2 MiB.")
        manifest = _json(archive.read("manifest.json"))
        if not isinstance(manifest, dict):
            return None
        if manifest.get("schema") != _SCHEMA:
            return None
        if len(names) != len(set(names)):
            raise ValueError("Combined bundle has duplicate member inventory entries.")
        if (len(infos) > 80 or
                any(item.is_dir() or item.flag_bits & 1 or
                    item.compress_type not in (zipfile.ZIP_STORED,
                                               zipfile.ZIP_DEFLATED) or
                    ((item.external_attr >> 16) & 0o170000) == 0o120000 or
                    item.file_size > 64 * 1024 * 1024 or
                    item.file_size > max(1024, 1000 * item.compress_size)
                    for item in infos) or
                sum(item.file_size for item in infos) > _LIMIT):
            raise ValueError("Combined bundle member inventory is unsafe.")
        expected = set(manifest.get("members", {}))
        if (set(names) - {"manifest.json"} != expected or
                any(name.startswith("/") or "\\" in name or
                    ".." in Path(name).parts for name in expected)):
            raise ValueError("Combined bundle member names differ from manifest.")
        members = {}
        for name in expected:
            data = archive.read(name)
            record = manifest["members"][name]
            if (not isinstance(record, dict) or record.get("bytes") != len(data) or
                    record.get("sha256") != hashlib.sha256(data).hexdigest()):
                raise ValueError("Combined bundle member hash mismatch: " + name)
            members[name] = data
    return manifest, members


def write_combined_scalar_archive(model, path):
    """Write a configured composite with compiler tables embedded and verified."""
    basis = model.basis
    if basis.source != "combined_scalar":
        raise TypeError("Expected a combined scalar model.")
    if "archive_bytes" in model._fitted:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="ye3t_combined_write_", dir=target.parent) as directory:
            temporary = Path(directory) / target.name
            temporary.write_bytes(model._fitted["archive_bytes"])
            temporary.replace(target)
        return target
    if any(not hasattr(item, "_construction") for item in basis._components.values()):
        raise ValueError("The safe combined writer requires independently configured components.")
    members = {}
    couplings = {}
    weights = {"per_species_E0_eV": np.asarray(
        [model._fitted["offsets_eV"][name] for name in basis.elements], dtype=np.float64)}
    representation = []
    sources = []
    component_records = []
    for index, (name, item) in enumerate(basis._components.items()):
        fitted = model._fitted["components"][name]
        if tuple(item.elements) != tuple(basis.elements):
            raise ValueError("Combined component species order changed before export.")
        construction = item._construction
        record = {"name": name, "kind": item.source,
                  "resolution_sha256": item._resolution.sha256,
                  "member": f"components/{index}.json"}
        component_records.append(record)
        representation.append({"name": name,
                               "input": construction["representation"],
                               "resolved": item._resolution.to_dict()["representation"]})
        sources.append({"name": name, "basis": construction["basis"],
                        "runtime": construction["runtime"],
                        "resolved_single_factors": item._resolution.to_dict()["single_factors"]})
        if item.source == "density":
            application = deepcopy(item._descriptor.ace_descriptor.ordinary_scalar_catalogue)
            if application is None or application.get("schema") != "ye3t_ordinary_scalar_catalogue_v2":
                raise ValueError("Configured density needs serialized v2 compiler coordinates.")
            record["compiler_hash"] = application["application_sha256"]
            for row_index, row in enumerate(application["rows"]):
                coordinate = row["compiled_coordinate"]
                prefix = f"c{index}_r{row_index}"
                couplings[prefix + "_m"] = np.asarray(
                    coordinate.pop("magnetic_tuples"), dtype=np.int64)
                couplings[prefix + "_c"] = np.asarray(
                    coordinate.pop("coefficients"), dtype=np.float64)
            members[record["member"]] = _json_bytes(application)
            weights[f"c{index}"] = np.asarray(fitted.weight, dtype=np.float64)
            if (weights[f"c{index}"].shape != (len(item.labels),) or
                    not np.isfinite(weights[f"c{index}"]).all() or fitted.bias != 0.0):
                raise ValueError("Combined density coefficients or offset are invalid.")
        else:
            record["compiler_hash"] = item._descriptor.metadata[
                "tagged_cauchy_image_compiled"].self_hash
            with tempfile.TemporaryDirectory(prefix="ye3t_combined_tagged_") as directory:
                target = Path(directory) / "tagged.ye3t.json"
                fitted.export_lammps(target)
                members[record["member"]] = target.read_bytes()
            weights[f"c{index}"] = np.concatenate(tuple(
                fitted.beta_by_species[element].detach().cpu().numpy()
                for element in basis.elements))
            if (weights[f"c{index}"].shape !=
                    (len(item.labels) * len(basis.elements),) or
                    not np.isfinite(weights[f"c{index}"]).all()):
                raise ValueError("Combined tagged coefficient width is invalid.")
    metadata = deepcopy(model._fitted["fit_metadata"])
    from .linear import _combined_design_columns
    if (metadata.get("design_columns") !=
            _combined_design_columns(basis, metadata.get("fit_E0")) or
            type(metadata.get("fit_E0")) is not bool or
            metadata.get("n_cols") != len(metadata["design_columns"]) or
            metadata.get("per_species_E0_eV") != model._fitted["offsets_eV"] or
            any(abs(sum(component.offsets[element]
                        for component in model._fitted["components"].values()
                        if hasattr(component, "offsets")) -
                    model._fitted["offsets_eV"][element]) > 1e-12
                for element in basis.elements)):
        raise ValueError("Combined fit columns or final offsets changed before export.")
    posterior = metadata.get("predictive_uncertainty")
    if posterior is not None:
        for field, key in (("coefficient_precision", "posterior_precision"),
                           ("coefficient_covariance_active", "posterior_covariance")):
            weights[key] = np.asarray(posterior.pop(field), dtype=np.float64)
            posterior[field] = {"array": key}
    members.update({
        "representation.json": _json_bytes(representation),
        "sources.json": _json_bytes(sources),
        "labels.json": _json_bytes([label.as_dict() for label in basis.labels]),
        "fit.json": _json_bytes(metadata),
        "couplings.npz": _npz_bytes(couplings),
        "weights.npz": _npz_bytes(weights),
    })
    identity = {"representation_sha256": hashlib.sha256(members["representation.json"]).hexdigest(),
                "sources_sha256": hashlib.sha256(members["sources.json"]).hexdigest(),
                "labels_sha256": hashlib.sha256(members["labels.json"]).hexdigest(),
                "compiler_hashes": [row["compiler_hash"] for row in component_records]}
    manifest = {"schema": _SCHEMA, "species_order": list(basis.elements),
                "output_type": "scalar_energy_forces_stress",
                "implementation_versions": {"combined_archive": 1,
                                            "ordinary_catalogue": 2,
                                            "tagged_cauchy_model": 1},
                "convention_sha256": hashlib.sha256(_json_bytes(identity)).hexdigest(),
                "provenance": {"compiler_api": "ye3t.couplings",
                               "source": "configured_components",
                               "fit_method": metadata["fit_method"]},
                "arrays": {"couplings.npz": _array_inventory(couplings),
                           "weights.npz": _array_inventory(weights)},
                "components": component_records,
                "members": {name: {"bytes": len(data),
                                   "sha256": hashlib.sha256(data).hexdigest()}
                            for name, data in sorted(members.items())}}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ye3t_combined_write_", dir=target.parent) as directory:
        temporary = Path(directory) / target.name
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr("manifest.json", _json_bytes(manifest))
            for name, data in sorted(members.items()):
                archive.writestr(name, data)
        temporary.replace(target)
    return target


def read_combined_scalar_archive(path):
    """Read only verified configured components and their indexed readout."""
    from .linear import Basis, _combined_design_columns
    from .config import resolve_linear_fit_config

    with Path(path).open("rb") as handle:
        raw = handle.read(_LIMIT + 1)
    verified = _verified_members(raw)
    if verified is None:
        return None
    manifest, members = verified
    records = manifest.get("components")
    saved_species = manifest.get("species_order")
    species = tuple(saved_species) if isinstance(saved_species, list) else ()
    if (not isinstance(records, list) or not 2 <= len(records) <= 16 or
            any(not isinstance(row, dict) for row in records) or
            not isinstance(species, tuple) or
            any(not isinstance(name, str) or not name for name in species) or
            any(not isinstance(row.get("name"), str) or not row["name"]
                for row in records) or
            len({row["name"] for row in records}) != len(records) or
            not species or len(set(species)) != len(species) or
            manifest.get("output_type") != "scalar_energy_forces_stress"):
        raise ValueError("Combined bundle component or species manifest is invalid.")
    required = {"representation.json", "sources.json", "labels.json",
                "fit.json", "couplings.npz", "weights.npz"}
    if set(members) != required | {f"components/{index}.json" for index in range(len(records))}:
        raise ValueError("Combined bundle has an unexpected member inventory.")
    representation = _json(members["representation.json"])
    sources = _json(members["sources.json"])
    labels = _json(members["labels.json"])
    metadata = _json(members["fit.json"])
    identity = {"representation_sha256": hashlib.sha256(members["representation.json"]).hexdigest(),
                "sources_sha256": hashlib.sha256(members["sources.json"]).hexdigest(),
                "labels_sha256": hashlib.sha256(members["labels.json"]).hexdigest(),
                "compiler_hashes": [row.get("compiler_hash") for row in records]}
    if (manifest.get("implementation_versions") !=
            {"combined_archive": 1, "ordinary_catalogue": 2,
             "tagged_cauchy_model": 1} or
            manifest.get("convention_sha256") !=
            hashlib.sha256(_json_bytes(identity)).hexdigest() or
            manifest.get("provenance") !=
            {"compiler_api": "ye3t.couplings",
             "source": "configured_components",
             "fit_method": metadata.get("fit_method")}):
        raise ValueError("Combined bundle version, convention, or provenance changed.")
    if (not isinstance(metadata, dict) or
            not isinstance(representation, list) or
            not isinstance(sources, list) or
            any(not isinstance(row, dict) for row in representation + sources) or
            len(representation) != len(records) or len(sources) != len(records)):
        raise ValueError("Combined bundle source and representation counts differ.")
    coupling_names = set()
    weight_names = {"per_species_E0_eV"}
    for index, record in enumerate(records):
        if record.get("member") != f"components/{index}.json" or record.get("kind") not in {
                "density", "tagged_cauchy_image"}:
            raise ValueError("Combined bundle component order or kind is invalid.")
        weight_names.add(f"c{index}")
        if record["kind"] == "density":
            application = _json(members[record["member"]])
            if application.get("application_sha256") != record.get("compiler_hash"):
                raise ValueError("Combined ordinary compiler identity changed.")
            for row_index in range(len(application.get("rows", ()))):
                prefix = f"c{index}_r{row_index}"
                coupling_names.update((prefix + "_m", prefix + "_c"))
    posterior = metadata.get("predictive_uncertainty")
    if posterior is not None:
        weight_names.update(("posterior_precision", "posterior_covariance"))
    coupling_arrays = _npz(members["couplings.npz"], coupling_names)
    weight_arrays = _npz(members["weights.npz"], weight_names)
    if manifest.get("arrays") != {"couplings.npz": _array_inventory(coupling_arrays),
                                  "weights.npz": _array_inventory(weight_arrays)}:
        raise ValueError("Combined bundle array dimensions or dtypes changed.")
    offsets = weight_arrays["per_species_E0_eV"]
    if offsets.shape != (len(species),):
        raise ValueError("Combined bundle E0 width differs from species order.")
    offset_map = dict(zip(species, map(float, offsets)))
    if posterior is not None:
        _validate_posterior(posterior, weight_arrays, metadata.get("n_cols"))
        for field, key in (("coefficient_precision", "posterior_precision"),
                           ("coefficient_covariance_active", "posterior_covariance")):
            if posterior.get(field) != {"array": key}:
                raise ValueError("Combined posterior array reference is invalid.")
            posterior[field] = weight_arrays[key].tolist()
    components = {}
    fitted = {}
    for index, record in enumerate(records):
        name = record["name"]
        if (not isinstance(name, str) or not name or
                representation[index].get("name") != name or
                sources[index].get("name") != name):
            raise ValueError("Combined bundle component identities differ.")
        if record["kind"] == "density":
            application = _json(members[record["member"]])
            for row_index, row in enumerate(application["rows"]):
                prefix = f"c{index}_r{row_index}"
                magnetic = coupling_arrays[prefix + "_m"]
                coefficient = coupling_arrays[prefix + "_c"]
                if (magnetic.ndim != 2 or coefficient.shape != (len(magnetic), 2) or
                        magnetic.shape[1] != len(row["compiled_coordinate"]["label"]["n_tuple"])):
                    raise ValueError("Combined ordinary coupling array shape is invalid.")
                row["compiled_coordinate"]["magnetic_tuples"] = magnetic.tolist()
                row["compiled_coordinate"]["coefficients"] = coefficient.tolist()
            source = sources[index]
            rep = YE3TRepresentation.from_config(representation[index]["input"])
            basis = Basis.from_config(source["basis"], representation=rep,
                                      runtime=source["runtime"],
                                      _check_optional_dependencies=False)
            if (basis._resolution.sha256 != record["resolution_sha256"] or
                    basis._resolution.to_dict()["representation"] !=
                    representation[index]["resolved"] or
                    basis._resolution.to_dict()["single_factors"] !=
                    source["resolved_single_factors"]):
                raise ValueError("Combined ordinary resolution identity changed.")
            basis._materialize_configured(ordinary_catalogue=application)
            weight = weight_arrays[f"c{index}"]
            if weight.shape != (len(basis.labels),):
                raise ValueError("Combined ordinary weight width differs from labels.")
            descriptor = basis._descriptor
            fitted[name] = LinearACEScalarModelBundle(
                settings=descriptor.settings, site_basis_config=descriptor.site_basis_config,
                descriptor_specs=descriptor.descriptor_specs, weight=weight,
                bias=0.0, basis_mode=None, fit_method="combined_archive",
                fit_metadata={"combined_component": name},
            )
        else:
            source = sources[index]
            rep = YE3TRepresentation.from_config(representation[index]["input"])
            resolved_basis = Basis.from_config(source["basis"], representation=rep,
                                               runtime=source["runtime"],
                                               _check_optional_dependencies=False)
            if (resolved_basis._resolution.sha256 != record["resolution_sha256"] or
                    resolved_basis._resolution.to_dict()["representation"] !=
                    representation[index]["resolved"] or
                    resolved_basis._resolution.to_dict()["single_factors"] !=
                    source["resolved_single_factors"]):
                raise ValueError("Combined tagged resolution identity changed.")
            with tempfile.TemporaryDirectory(prefix="ye3t_combined_read_") as directory:
                target = Path(directory) / "tagged.ye3t.json"
                target.write_bytes(members[record["member"]])
                tagged = load_tagged_cauchy_image_model(target,
                                                        compiler_validation="certificate")
            basis = Basis._from_tagged_model(tagged)
            basis._loaded_tagged_model = tagged
            basis._resolution = resolved_basis._resolution
            basis._construction = resolved_basis._construction
            basis._catalogue = resolved_basis._catalogue
            basis._resolved = resolved_basis._resolved
            radial = source["resolved_single_factors"]["radial"]
            if (tuple(tagged.species_order) != species or
                    radial["family"] != "shifted_jacobi" or
                    float(radial["cutoff_A"]) != float(tagged.evaluator.cutoff) or
                    tagged.evaluator.compiled.self_hash != record.get("compiler_hash")):
                raise ValueError("Combined tagged source or compiler identity changed.")
            saved_beta = weight_arrays[f"c{index}"]
            actual_beta = np.concatenate(tuple(
                tagged.beta_by_species[element].detach().cpu().numpy()
                for element in species))
            if saved_beta.shape != actual_beta.shape or not np.array_equal(saved_beta, actual_beta):
                raise ValueError("Combined tagged weights differ from component artifact.")
            fitted[name] = tagged
        components[name] = basis
    basis = Basis.combine(components)
    if (tuple(basis.elements) != species or
            _json_bytes([label.as_dict() for label in basis.labels]) != _json_bytes(labels) or
            metadata.get("per_species_E0_eV") != offset_map or
            type(metadata.get("fit_E0")) is not bool or
            metadata.get("design_column_order") !=
            "component_features_then_species_offsets" or
            metadata.get("design_columns") !=
            _combined_design_columns(basis, metadata["fit_E0"]) or
            metadata.get("n_cols") != len(metadata["design_columns"])):
        raise ValueError("Combined bundle labels or final offsets changed.")
    fit_record = metadata.get("resolved_fit_config")
    fit_hash = metadata.get("resolved_fit_config_sha256")
    if fit_record is not None or fit_hash is not None:
        if (not isinstance(fit_record, dict) or
                fit_record.get("schema") != "ye3t_configured_scalar_fit_v1" or
                fit_hash != hashlib.sha256(_json_bytes(fit_record)).hexdigest() or
                not isinstance(fit_record.get("basis_resolution"), dict) or
                fit_record.get("basis_resolution_sha256") != hashlib.sha256(
                    _json_bytes(fit_record["basis_resolution"])).hexdigest()):
            raise ValueError("Combined configured fit identity changed.")
        resolved = fit_record["basis_resolution"]
        fit_model = fit_record.get("model")
        fit_reference = fit_model.get("reference_energy") if isinstance(fit_model, dict) else None
        construction = fit_record.get("construction")
        if (resolved.get("species") != list(species) or
                not isinstance(resolved.get("components"), list) or
                len(resolved["components"]) != len(components) or
                any(not isinstance(row, dict) or row.get("name") != name or
                    row.get("single_factors") != components[name]._resolution.to_dict()[
                        "single_factors"]
                    for row, name in zip(resolved["components"], components)) or
                not isinstance(construction, dict) or
                set(construction) != {"representation", "basis", "runtime"} or
                not isinstance(fit_reference, dict) or
                fit_reference.get("fit_E0") != metadata["fit_E0"]):
            raise ValueError("Combined configured fit source or E0 policy changed.")
        try:
            rep = YE3TRepresentation.from_config(construction["representation"])
            configured = Basis.from_config(construction["basis"], representation=rep,
                                           runtime=construction["runtime"],
                                           _check_optional_dependencies=False)
            full = {"metadata": fit_record["metadata"],
                    **construction, "model": fit_record["model"],
                    "targets": fit_record["targets"],
                    "validation": fit_record["validation"]}
            resolved_fit = resolve_linear_fit_config(
                full, configured, check_optional_dependencies=False)
            source_components = construction["basis"]["components"]
            component_hashes = {}
            for name in components:
                raw_component = source_components[name]
                local = {
                    "single_factors": deepcopy(raw_component.get(
                        "single_factors", construction["basis"]["single_factors"])),
                    "tensor_product": raw_component["tensor_product"],
                    "catalogue": raw_component["catalogue"],
                }
                local["single_factors"]["species"] = list(configured.elements)
                component_hashes[name] = Basis.from_config(
                    local, representation=rep,
                    runtime=construction["runtime"],
                    _check_optional_dependencies=False).resolution.sha256
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise ValueError("Combined configured fit construction is invalid.") from error
        if (resolved_fit["resolved_fit_config"] != fit_record or
                resolved_fit["resolved_fit_config_sha256"] != fit_hash or
                configured.resolution.sha256 != fit_record["basis_resolution_sha256"] or
                any(component_hashes[name] != components[name]._resolution.sha256
                    for name in components) or
                resolved_fit["method"] != metadata.get("fit_method") or
                (resolved_fit["method"] == "ridge" and
                 resolved_fit["regularization"] != metadata.get("alpha")) or
                (resolved_fit["method"] != "ridge" and
                 resolved_fit["sklearn_params"] != metadata.get("sklearn_params")) or
                (not metadata["fit_E0"] and
                 resolved_fit["reference_energies"] != offset_map) or
                metadata.get("configured_validation", {}).get("checks") !=
                list(resolved_fit["validation_checks"])):
            raise ValueError("Combined configured fit semantics differ from the artifact.")
    tagged_offsets = [model.offsets for model in fitted.values()
                      if hasattr(model, "offsets")]
    if any(abs(sum(item[name] for item in tagged_offsets) - offset_map[name]) > 1e-12
           for name in species):
        raise ValueError("Combined bundle applies a species offset more than once.")
    return {"basis": basis, "fitted": {"components": fitted,
                                       "offsets_eV": offset_map,
                                       "fit_metadata": metadata,
                                       "archive_bytes": raw}}
