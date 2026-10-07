"""Bounded, compiler-bound Ni scalar bundle reader and Torch evaluator.

The v1 reader consumes JSON/NPZ model members. It never reads legacy
compatibility files or compiles coupling coefficients during model loading.
"""

from io import BytesIO
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.mixing import SumCalculator

from ye3t.couplings import normalize_compact_label
from ye3t.execution_plan import compile_tagged_moment_execution_portfolio
from ye3t_methods.atomistic import YE3TDescriptors, YE3TRepresentation
from ye3t_methods.atomistic.ace.linear_ace import (
    LinearACEScalarCalculator, LinearACEScalarModelBundle, _linear_ace_geometry_row,
)
from ye3t_methods.atomistic.reference_potentials import YE3TZBLCalculator
from ye3t_methods.atomistic.tagged_cauchy_linear import (
    TaggedCauchyModel, ordinary_edge_primitives,
    ordinary_edge_primitives_with_derivative,
)


_CORE = {
    "representation.json", "sources.json", "labels.json",
    "couplings.npz", "weights.npz", "native_plan.json",
    "source/ordinary_catalogue.json", "source/tagged_catalogue.json",
    "source/tagged_runtime.json", "source/tagged_pooled_program.json",
}
_NI_PROFILES = {
    "Ni/ye3t_tagged_60": (15, 45, 60,
                           "14b8db7ec630208fb37818c6bc56fd626c08759d7583c8fe7fe427bd7f387443"),
    "Ni/ye3t_tagged_127": (59, 68, 127,
                            "a57647406108a71273794e7786252147c93830d8954d8c44d9d7e813cb6502a2"),
    "Ni/ye3t_tagged_149": (81, 68, 149,
                            "c6729f00ab2c928d2c9969f9083954eda872e74a16249685efc975820f8fa49b"),
    "Ni/ye3t_augmented_196": (128, 68, 196,
                               "5ff1a662d8951a35e7894642226c73ad836f8ea139c9da99804c67dfbad4367c"),
}
_NI_PROGRAM_SHA256 = "c93b28c5790c7b8fc624ccbe3101dbaf507fa917e2003bec90557e9ed35bef4b"


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _pairs(rows):
    result = {}
    for key, value in rows:
        if key in result:
            raise ValueError("Portable bundle JSON has a duplicate key.")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("Portable bundle JSON has a nonfinite constant: " + value)


def _finite_float(value):
    number = float(value)
    if not np.isfinite(number):
        raise ValueError("Portable bundle JSON has a nonfinite number.")
    return number


def _json(payload):
    return json.loads(payload.decode("utf-8"), object_pairs_hook=_pairs,
                      parse_constant=_reject_constant, parse_float=_finite_float)


def _arrays(payload, required):
    with zipfile.ZipFile(BytesIO(payload)) as inner:
        infos = inner.infolist()
        if (not infos or len(infos) != len(required) or
                {item.filename for item in infos} != {key + ".npy" for key in required} or
                any(item.file_size > 16 * 1024 * 1024 or item.flag_bits & 1
                    for item in infos) or
                sum(item.file_size for item in infos) > 64 * 1024 * 1024):
            raise ValueError("Portable bundle NPZ has an invalid bounded inventory.")
    with np.load(BytesIO(payload), allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in required}
    if any(array.dtype not in (np.dtype("int64"), np.dtype("float64")) or
           array.ndim != 1 or
           (array.dtype == np.dtype("float64") and not np.isfinite(array).all())
           for array in arrays.values()):
        raise ValueError("Portable bundle arrays must be finite 1-D int64/float64 data.")
    return arrays


def _read_members(raw):
    if len(raw) > 256 * 1024 * 1024:
        raise ValueError("Portable bundle exceeds the 256 MiB file limit.")
    with zipfile.ZipFile(BytesIO(raw)) as archive:
        infos = archive.infolist()
        by_name = {item.filename: item for item in infos}
        if "manifest.json" not in by_name or len(infos) != len(by_name):
            raise ValueError("Portable bundle has duplicate or missing members.")
        manifest_info = by_name["manifest.json"]
        if manifest_info.file_size > 2 * 1024 * 1024:
            raise ValueError("Portable bundle manifest exceeds 2 MiB.")
        manifest = _json(archive.read("manifest.json"))
        if manifest.get("schema") != "ye3t_linear_portable_bundle_v1":
            return None
        if (manifest.get("maturity") != "bounded_ni_scalar_candidate" or
                manifest.get("inference_route") != "portable_members_only" or
                manifest.get("model") not in _NI_PROFILES or
                manifest.get("species_order") != ["Ni"] or
                manifest.get("output_type") != "energy_forces_stress"):
            raise ValueError("Unsupported portable scalar bundle contract.")
        names = set(by_name) - {"manifest.json"}
        if (len(infos) > 64 or not _CORE.issubset(names) or
                len([name for name in names if name.startswith(
                    "source/tagged_artifacts/")]) != 11 or
                len([name for name in names if name.startswith(
                    "source/tagged_images/")]) != 11 or
                names != set(manifest.get("members", {})) or
                any(name.startswith("compat/") or name.startswith("/") or
                    "\\" in name or ".." in Path(name).parts for name in names) or
                any(item.is_dir() or item.flag_bits & 1 or
                    item.compress_type not in (zipfile.ZIP_STORED,
                                               zipfile.ZIP_DEFLATED) or
                    ((item.external_attr >> 16) & 0o170000) == 0o120000 or
                    item.file_size > 64 * 1024 * 1024 or
                    item.file_size > max(1024, 1000 * item.compress_size)
                    for item in infos) or
                sum(item.file_size for item in infos) > 256 * 1024 * 1024):
            raise ValueError("Portable bundle member inventory is unsafe or incomplete.")
        members = {}
        for name in names:
            data = archive.read(name)
            record = manifest["members"][name]
            if (not isinstance(record, dict) or
                    record.get("bytes") != len(data) or
                    record.get("sha256") != hashlib.sha256(data).hexdigest()):
                raise ValueError("Portable bundle member hash mismatch: " + name)
            members[name] = data
    return manifest, members


def read_portable_linear_archive(path):
    """Return a verified bounded Ni record, or None for another .ye3t schema."""
    if isinstance(path, bytes):
        raw = path
    else:
        target = Path(path)
        with target.open("rb") as handle:
            raw = handle.read(256 * 1024 * 1024 + 1)
    if len(raw) > 256 * 1024 * 1024:
        raise ValueError("Portable bundle exceeds the 256 MiB file limit.")
    with zipfile.ZipFile(BytesIO(raw)) as archive:
        if "manifest.json" not in archive.namelist():
            return None
        info = archive.getinfo("manifest.json")
        if info.file_size > 2 * 1024 * 1024:
            raise ValueError("Portable bundle manifest exceeds 2 MiB.")
        preview = _json(archive.read("manifest.json"))
    if not isinstance(preview, dict) or preview.get("schema") != "ye3t_linear_portable_bundle_v1":
        return None
    profile = _NI_PROFILES.get(preview.get("model"))
    if profile is None or hashlib.sha256(raw).hexdigest() != profile[3]:
        raise ValueError("Portable Ni archive trusted SHA-256 mismatch.")
    opened = _read_members(raw)
    if opened is None:
        return None
    manifest, members = opened
    records = {name: _json(members[name]) for name in members
               if name.endswith(".json")}
    weights = _arrays(members["weights.npz"], {
        "feature_indices", "ordinary", "tagged_selected", "tagged_beta_69",
        "per_species_E0_Ni"})
    couplings = _arrays(members["couplings.npz"], {
        "ordinary_rank", "ordinary_row_offsets", "ordinary_magnetic_flat",
        "ordinary_coefficient_real", "ordinary_coefficient_imag",
        "tagged_sparse_offsets", "tagged_descriptor_index",
        "tagged_image_real", "tagged_image_imag"})
    source = records["sources.json"]
    catalogue = records["source/ordinary_catalogue.json"]
    runtime = records["source/tagged_runtime.json"]
    pooled = records["source/tagged_pooled_program.json"]
    representation = records["representation.json"]
    labels = records["labels.json"]["ordered"]
    selected = weights["feature_indices"].tolist()
    ordinary_count = len(weights["ordinary"])
    tagged_count = len(weights["tagged_selected"])
    expected_ordinary, expected_tagged, width, _ = profile
    if (ordinary_count != expected_ordinary or tagged_count != expected_tagged or
            len(labels) != width or
            len(selected) != len(labels) or len(set(selected)) != len(selected) or
            [row["fit_column"] for row in labels] != list(range(width)) or
            len(weights["tagged_beta_69"]) != 69 or
            len(weights["per_species_E0_Ni"]) != 1 or
            len(catalogue["rows"]) != 149 or
            len(runtime["features"]) != 69):
        raise ValueError("Portable Ni-127 readout or catalogue width differs.")
    if (catalogue["membership_sha256"] != _hash({
            "profile_id": catalogue["profile_id"],
            "feature_ids": catalogue["feature_ids"]}) or
            catalogue["application_sha256"] != _hash({
                key: catalogue[key] for key in
                ("schema", "profile_id", "feature_ids", "rows", "compiler")}) or
            catalogue["feature_ids"] != [row["feature_id"]
                                         for row in catalogue["rows"]]):
        raise ValueError("Portable ordinary catalogue hash or row order differs.")
    if (source["species_order"] != ["Ni"] or
            representation["species_order"] != ["Ni"] or
            source["radial_fit"]["cutoff_A"] !=
            source["tagged_radial_definition"]["parameters"]["rc"] or
            source["target_operation"] != "ab_initio_minus_reference"):
        raise ValueError("Portable Ni physical source association differs.")
    YE3TZBLCalculator.from_model_manifest(
        {"reference_potential": source["reference_potential"]})
    ordinary_labels = tuple(normalize_compact_label(row["compact_label"])
                            for row in catalogue["rows"])
    ranks = tuple(sorted({int(label.rank) for label in ordinary_labels}))
    nmax = tuple(max(max(int(value) for value in label.n_tuple)
                     for label in ordinary_labels if int(label.rank) == rank)
                 for rank in ranks)
    lmax = tuple(max(max(int(value) for value in label.l_tuple)
                     for label in ordinary_labels if int(label.rank) == rank)
                 for rank in ranks)
    radial = source["radial_fit"]
    site = source["ordinary_site_basis_config"]
    expected_site = {
        "rc": [radial["cutoff_A"]], "lmbda": [radial["radial_lambda"]],
        "nradmax": max(nmax), "lmax": max(lmax), "kmax": 0,
        "possible_types": [0], "radial_basis": "PACE_ChebExpCos",
        "chemical_basis": "delta", "charge_mode": "none",
        "atomic_base_normalization": "none", "factor_normalization": "none",
        "spherical_backend": "complex", "spherical_normalization": "pace_y00_one",
        "source_backend": "torch", "dtype": "float64",
        "pace_cutoff_width": [radial["cutoff_width_A"]],
        "pace_spline_spacing": [0.001], "pace_inner_cutoff": [0.0],
        "pace_inner_cutoff_width": [0.0], "pace_crad_policy": "identity",
    }
    if (source["ordinary_tensor_orders"] != list(ranks) or
            source["ordinary_nmax_by_order"] != list(nmax) or
            source["ordinary_lmax_by_order"] != list(lmax) or
            site != expected_site):
        raise ValueError("Portable ordinary physical source contract differs.")
    offsets = couplings["ordinary_row_offsets"]
    ranks_array = couplings["ordinary_rank"]
    real = couplings["ordinary_coefficient_real"]
    imag = couplings["ordinary_coefficient_imag"]
    magnetic = couplings["ordinary_magnetic_flat"]
    if (len(offsets) != ordinary_count + 1 or offsets[0] != 0 or
            np.any(np.diff(offsets) <= 0) or offsets[-1] != len(real) or
            len(real) != len(imag) or
            len(magnetic) != int(np.dot(np.diff(offsets), ranks_array))):
        raise ValueError("Portable ordinary coefficient array shape differs.")
    magnetic_at = 0
    for j, catalogue_index in enumerate(selected[:ordinary_count]):
        if not 0 <= catalogue_index < 149:
            raise ValueError("Portable ordinary selection is outside the catalogue.")
        row = catalogue["rows"][catalogue_index]
        coordinate = row["compiled_coordinate"]
        certificate = coordinate["certificate"]
        label = labels[j]
        start, stop = offsets[j:j + 2]
        rank = len(coordinate["label"]["n_tuple"])
        if (label["branch"] != "ordinary" or
                label["catalogue_index"] != catalogue_index or
                label["feature_id"] != row["feature_id"] or
                label["label"] != coordinate["label"] == row["compact_label"] or
                label["coordinate_payload_sha256"] != coordinate["payload_sha256"] or
                coordinate["payload_sha256"] != _hash({
                    key: value for key, value in coordinate.items()
                    if key != "payload_sha256"}) or
                certificate.get("passed") is not True or
                certificate["certificate_sha256"] != _hash({
                    key: value for key, value in certificate.items()
                    if key != "certificate_sha256"}) or
                ranks_array[j] != rank or stop - start !=
                len(coordinate["coefficients"]) or
                magnetic[magnetic_at:magnetic_at + rank * (stop - start)].tolist()
                != [m for row_m in coordinate["magnetic_tuples"] for m in row_m] or
                not np.array_equal(real[start:stop], np.asarray(
                    coordinate["coefficients"])[:, 0]) or
                not np.array_equal(imag[start:stop], np.asarray(
                    coordinate["coefficients"])[:, 1])):
            raise ValueError("Portable ordinary compiler label/coordinate binding differs.")
        magnetic_at += rank * (stop - start)
    if (not all(row["branch"] == "tagged" for row in labels[ordinary_count:]) or
            [row["tagged_program_column"] for row in labels[ordinary_count:]] !=
            [index - 149 for index in selected[ordinary_count:]] or
            any(not 149 <= index < 218 for index in selected[ordinary_count:])):
        raise ValueError("Portable tagged fit column order differs.")
    tagged_offsets = couplings["tagged_sparse_offsets"]
    tagged_indices = couplings["tagged_descriptor_index"]
    tagged_real = couplings["tagged_image_real"]
    tagged_imag = couplings["tagged_image_imag"]
    if (len(tagged_offsets) != tagged_count + 1 or tagged_offsets[0] != 0 or
            np.any(np.diff(tagged_offsets) <= 0) or
            tagged_offsets[-1] != len(tagged_indices) == len(tagged_real)
            == len(tagged_imag)):
        raise ValueError("Portable tagged sparse image array shape differs.")
    contents = {item["content_index"]: item for item in runtime["contents"]}
    for j, row in enumerate(labels[ordinary_count:]):
        feature = runtime["features"][row["tagged_program_column"]]
        content = contents[feature["content_index"]]
        artifact = records["source/tagged_artifacts/" + content["request_hash"] +
                           ".json"]
        combination = feature["combination"]
        start, stop = tagged_offsets[j:j + 2]
        if (row["component_index"] != feature["content_index"] or
                row["compiler_request_hash"] != content["request_hash"] or
                row["compiler_artifact_self_hash"] != artifact["self_hash"] or
                row["descriptor_indices"] != [item[0] for item in combination] or
                row["descriptor_labels"] != [artifact["plan"]["report"]["labels"][
                    index] for index, _ in combination] or
                stop - start != len(combination) or
                tagged_indices[start:stop].tolist() != [item[0]
                                                       for item in combination] or
                not np.array_equal(tagged_real[start:stop], np.asarray(
                    [item[1][0] for item in combination])) or
                not np.array_equal(tagged_imag[start:stop], np.asarray(
                    [item[1][1] for item in combination])) or
                weights["tagged_beta_69"][row["tagged_program_column"]] !=
                weights["tagged_selected"][j]):
            raise ValueError("Portable tagged compiler/fit binding differs.")
    selected_tagged = tuple(row["tagged_program_column"]
                            for row in labels[ordinary_count:])
    tagged_catalogue = records["source/tagged_catalogue.json"]
    components = tagged_catalogue["components"]
    runtime_contents = runtime["contents"]
    requests = [item["request_hash"] for item in components]
    if (len(components) != 11 or len(runtime_contents) != 11 or
            len(set(requests)) != 11 or
            {item["content_index"] for item in runtime_contents} !=
            {item["component_index"] for item in components} or
            {name for name in members if name.startswith("source/")} !=
            ({"source/ordinary_catalogue.json", "source/tagged_catalogue.json",
              "source/tagged_runtime.json", "source/tagged_pooled_program.json"} |
             {"source/tagged_artifacts/" + request + ".json" for request in requests} |
             {"source/tagged_images/" + request + ".json" for request in requests})):
        raise ValueError("Portable tagged compiler/image inventory differs.")
    feature_start = 0
    by_content = {item["content_index"]: item for item in runtime_contents}
    for component in components:
        content = by_content[component["component_index"]]
        request = component["request_hash"]
        artifact = records["source/tagged_artifacts/" + request + ".json"]
        image = records["source/tagged_images/" + request + ".json"]
        normalized = dict(content["request_payload"])
        families = normalized.pop("families")
        normalized["family_specs"] = families
        normalized["family_ids"] = [family["id"] for family in families]
        normalized["channels"] = [
            {key: value for key, value in channel.items() if key != "channel_index"}
            for channel in normalized["channels"]]
        if (content["request_hash"] != request or _hash(normalized) != request or
                artifact["self_hash"] != _hash({
                    key: value for key, value in artifact.items()
                    if key != "self_hash"}) or
                artifact["self_hash"] != content["artifact_self_hash"] or
                artifact["self_hash"] != component["artifact_self_hash"] or
                artifact["plan"]["report"]["request"] != content["request_payload"] or
                artifact["plan"]["provenance"]["api"] != "ye3t.couplings.plan" or
                image["record_hash"] != _hash({
                    key: value for key, value in image.items()
                    if key != "record_hash"}) or
                image["role_bindings"] != runtime["spec"]["role_bindings"] or
                image["raw_descriptor_count"] !=
                len(artifact["plan"]["report"]["labels"]) or
                [item["label"] for item in artifact["payload"]["descriptors"]] !=
                artifact["plan"]["report"]["labels"] or
                image["independent_feature_count"] != len(image["image_from_raw"]) or
                image["source_product_certificate"] != "required_from_application" or
                image["certificates"]["physical_source_independence"] is not False or
                any(image["certificates"].get(key) is not True for key in (
                    "passed", "exact_arithmetic", "reconstruction_passed",
                    "residual_is_zero"))):
            raise ValueError("Portable tagged compiler/image association differs.")
        for matrix_row in image["image_from_raw"]:
            if feature_start >= len(runtime["features"]):
                raise ValueError("Portable tagged image overruns runtime columns.")
            feature = runtime["features"][feature_start]
            combination = [[index, value["binary64"]]
                           for index, value in enumerate(matrix_row)
                           if value["binary64"] != [0.0, 0.0]]
            if (len(matrix_row) != image["raw_descriptor_count"] or
                    feature["feature_index"] != feature_start or
                    feature["content_index"] != content["content_index"] or
                    feature["combination"] != combination or
                    any(value["binary64"][1] != 0.0 for value in matrix_row)):
                raise ValueError("Portable tagged image/runtime column association differs.")
            feature_start += 1
    if feature_start != len(runtime["features"]):
        raise ValueError("Portable tagged image coverage is incomplete.")
    if (np.count_nonzero(weights["tagged_beta_69"][list(
            set(range(69)) - set(selected_tagged))]) != 0 or
            pooled["program"]["feature_count"] != 69 or
            runtime["catalogue_hash"] != pooled["catalogue_hash"] or
            representation["tagged_real_form_records"] is None):
        raise ValueError("Portable tagged pooled program binding differs.")
    plan = records["native_plan.json"]
    if (_hash(pooled["program"]) != _NI_PROGRAM_SHA256 or
            plan["program_hash"] != _NI_PROGRAM_SHA256 or
            plan["portfolio_hash"] != _hash({key: value for key, value in plan.items()
                                        if key != "portfolio_hash"}) or
            _hash(plan) != _hash(compile_tagged_moment_execution_portfolio(
                pooled["program"], {"Ni": weights["tagged_beta_69"].tolist()}))):
        raise ValueError("Portable tagged plan differs from program/readout.")
    representation_object = YE3TRepresentation.ace(
        basis_mode=None, fast_path_policy="disable",
        metadata={"global_young_sector": "(N)",
                  "basis_convention": "pace_complex_magnetic_y00_1"})
    descriptor = YE3TDescriptors.ace({
        "elements": ["Ni"], "type_map": {"Ni": 0},
        "cutoff": radial["cutoff_A"], "ranks": ranks,
        "basis_type": "no_charge", "k_o_max": 0,
        "k_max": [0] * len(ranks), "nmax": nmax, "lmax": lmax,
        "lmin": [0] * len(ranks), "L_R": 0, "M_R_values": [0],
        "ordinary_scalar_catalogue": catalogue,
        "use_descriptor_cache": False,
        "factorized_descriptor_runtime_policy": "disable",
        "site_basis_config": site, "representation": representation_object,
        "backend": "pytorch", "strict_backend": True,
        "validate_backend": True, "device": "cpu",
    })
    ordinary_indices = tuple(selected[:ordinary_count])
    specs = tuple(descriptor.descriptor_specs[index] for index in ordinary_indices)
    if [spec.label.to_dict() for spec in specs] != [row["label"]
                                                    for row in labels[:ordinary_count]]:
        raise ValueError("Portable ordinary descriptor axes differ from labels.")
    bundle = LinearACEScalarModelBundle(
        settings=descriptor.settings, site_basis_config=descriptor.site_basis_config,
        descriptor_specs=specs, weight=weights["ordinary"],
        bias=float(weights["per_species_E0_Ni"][0]), basis_mode=None,
        fit_method="portable_saved_readout",
        fit_metadata={"factorized_descriptor_runtime_policy": "disable"})
    tagged_radial = source["tagged_radial_definition"]["parameters"]
    tagged_model = TaggedCauchyModel(
        compiled=None, role_bindings=runtime["spec"]["role_bindings"],
        descriptor_selection=None, beta=weights["tagged_beta_69"],
        offsets={"Ni": 0.0},
        radial_config={"lmbda": tagged_radial["lmbda"],
                       "cutoff_width": tagged_radial["cutoff_width"]},
        cutoff=float(tagged_radial["rc"]), species_order=("Ni",),
        channels=runtime["channels"], real_moment_program=pooled["program"],
        channel_real_form_ids=pooled["channel_real_form_ids"],
        real_form_records=representation["tagged_real_form_records"])
    return {
        "manifest": manifest, "labels": labels, "sources": source,
        "weights": weights, "ordinary_descriptor": descriptor,
        "ordinary_bundle": bundle, "ordinary_indices": ordinary_indices,
        "tagged_model": tagged_model, "tagged_indices": selected_tagged,
        "archive_bytes": raw,
    }


def portable_feature_rows(record, atoms):
    ordinary = np.asarray(record["ordinary_descriptor"].create(atoms),
                          dtype=np.float64)[:, record["ordinary_indices"]]
    model = record["tagged_model"]
    positions = torch.as_tensor(np.asarray(atoms.positions, dtype=np.float64))
    atom_types = torch.as_tensor([model.species_index[name] for name in
                                  atoms.get_chemical_symbols()], dtype=torch.long)
    cell = (torch.as_tensor(np.asarray(atoms.cell.array, dtype=np.float64))
            if atoms.cell.rank == 3 else None)
    pbc = tuple(bool(value) for value in atoms.pbc) if cell is not None else None
    with torch.no_grad():
        primitives = ordinary_edge_primitives(
            positions, atom_types, cell, pbc, model.cutoff,
            model.radial_config, model.channels)
        tagged = model.real_evaluator.descriptors(
            positions, atom_types, cell, pbc, primitives).detach().cpu().numpy()
    rows = np.column_stack((ordinary, tagged[:, record["tagged_indices"]]))
    if rows.shape != (len(atoms), len(record["labels"])) or not np.isfinite(rows).all():
        raise RuntimeError("Portable scalar descriptor rows are invalid.")
    return rows


def portable_feature_design_row(record, atoms):
    """Return selected Ni energy and force design rows in saved column order."""
    if any(atoms.pbc) and atoms.cell.rank != 3:
        raise ValueError("Periodic selected Ni design rows require a full-rank cell.")
    descriptor = record["ordinary_descriptor"]
    ordinary = _linear_ace_geometry_row(
        atoms, evaluator=descriptor.ace_descriptor.calculator.evaluator,
        descriptors=record["ordinary_bundle"].descriptor_specs,
        cutoff=record["sources"]["radial_fit"]["cutoff_A"], type_map={"Ni": 0},
        device="cpu", forces=True, stress=False, chunk_size=None,
    )
    model = record["tagged_model"]
    positions = torch.as_tensor(np.asarray(atoms.positions, dtype=np.float64))
    atom_types = torch.as_tensor(
        [model.species_index[name] for name in atoms.get_chemical_symbols()],
        dtype=torch.long,
    )
    cell = (torch.as_tensor(np.asarray(atoms.cell.array, dtype=np.float64))
            if atoms.cell.rank == 3 else None)
    pbc = tuple(bool(value) for value in atoms.pbc) if cell is not None else None
    primitives = ordinary_edge_primitives_with_derivative(
        positions, atom_types, cell, pbc, model.cutoff,
        model.radial_config, model.channels,
    )
    tagged, tagged_jacobian = model.real_evaluator.descriptors_and_jacobian(
        positions, atom_types, primitives, term_chunk_size=4096,
    )
    selected = list(record["tagged_indices"])
    tagged_sites = tagged[:, selected].detach().cpu().numpy()
    tagged_forces = (-tagged_jacobian[selected].reshape(len(selected), -1)
                     .T.detach().cpu().numpy())
    ordinary_sites = ordinary["site_features"].detach().cpu().numpy()
    sites = np.column_stack((ordinary_sites, tagged_sites))
    forces = np.column_stack((
        ordinary["forces"].detach().cpu().numpy(), tagged_forces,
    ))
    if (sites.shape != (len(atoms), len(record["labels"])) or
            forces.shape != (3 * len(atoms), len(record["labels"])) or
            not np.isfinite(sites).all() or not np.isfinite(forces).all()):
        raise RuntimeError("Portable Ni selected energy/force design row is invalid.")
    return {"site_features": sites, "energy": sites.sum(axis=0), "forces": forces}


class _PortableTaggedCalculator(Calculator):
    implemented_properties = ["energy", "energies", "forces", "stress"]

    def __init__(self, model):
        super().__init__()
        self.model = model

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        current = self.atoms
        model = self.model
        volume = float(current.get_volume()) if current.cell.rank == 3 else 0.0
        if volume > 0.0:
            cell0 = torch.as_tensor(np.asarray(current.cell.array, dtype=np.float64))
            cell = cell0.detach().clone().requires_grad_(True)
            scaled = torch.as_tensor(np.asarray(current.positions, dtype=np.float64)) @ (
                torch.linalg.inv(cell0))
            positions = scaled @ cell
            pbc = tuple(bool(value) for value in current.pbc)
        else:
            if "stress" in properties:
                raise ValueError("Portable tagged stress requires a positive-volume cell.")
            cell0 = cell = None
            positions = torch.tensor(np.asarray(current.positions, dtype=np.float64),
                                     requires_grad=True)
            pbc = None
        atom_types = torch.as_tensor([model.species_index[name] for name in
                                      current.get_chemical_symbols()], dtype=torch.long)
        primitives = ordinary_edge_primitives(
            positions, atom_types, cell, pbc, model.cutoff,
            model.radial_config, model.channels)
        per_atom = model.real_evaluator.descriptors(
            positions, atom_types, cell, pbc, primitives)
        atomic = per_atom @ model.beta
        energy = atomic.sum()
        if cell is None:
            (position_gradient,) = torch.autograd.grad(energy, (positions,))
        else:
            position_gradient, cell_gradient = torch.autograd.grad(
                energy, (positions, cell))
        self.results = {
            "energy": float(energy.detach()),
            "energies": atomic.detach().cpu().numpy(),
            "forces": -position_gradient.detach().cpu().numpy(),
        }
        if cell is not None:
            strain_gradient = cell0.T @ cell_gradient
            tensor = 0.5 * (strain_gradient + strain_gradient.T) / volume
            self.results["stress"] = tensor[[0, 1, 2, 1, 0, 0],
                                             [0, 1, 2, 2, 2, 1]].detach().cpu().numpy()


def portable_torch_calculator(record, neighbors="auto", **kwargs):
    if neighbors not in ("auto", "ase"):
        raise ValueError("Portable Torch scalar ASE supports auto or ase neighbors.")
    ordinary = LinearACEScalarCalculator(
        record["ordinary_bundle"], record["sources"]["radial_fit"]["cutoff_A"],
        {"Ni": 0}, backend="pytorch", strict_backend=True,
        factorized_descriptor_runtime_policy="disable", **kwargs)
    tagged = _PortableTaggedCalculator(record["tagged_model"])
    zbl = YE3TZBLCalculator.from_model_manifest(
        {"reference_potential": record["sources"]["reference_potential"]})
    return SumCalculator((ordinary, tagged, zbl))
